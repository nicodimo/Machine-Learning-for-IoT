import numpy as np
import sounddevice as sd
import redis
import argparse
import psutil
import uuid
import tensorflow as tf
from datetime import datetime
from time import sleep, time
from math import floor
import zipfile
import os

REDIS_HOST = 'redis-11747.c300.eu-central-1-1.ec2.cloud.redislabs.com'
REDIS_PORT = 11747
REDIS_PASSWORD = 'i4Wds8wGEcQ5PVQwBMmEQL17GH4XdXKH'

PLAIN_MEMORY_LIMIT = 5 * 1024 * 1024
AGGREGATED_MEMORY_LIMIT = 1024 * 1024

LABELS = ['go', 'stop']
PREPROCESSING_ARGS = {
    'downsampling_rate': 16000,
    'frame_lenght_in_s': 0.032,
    'frame_step_in_s': 0.032,
    'num_mel_bins': 25,
    'lower_f': 0,
    'upper_f': 8000,
    'num_coefficients': 13
}

MODEL_PATH = './model17.tflite.zip'
MODEL_NAME = 'model17'

keyword = 'go'


def get_audio_from_numpy(indata):
    indata = tf.convert_to_tensor(indata, dtype=tf.float32)
    indata = 2 * ((indata + 32768) / (32767 + 32768)) - 1
    indata = tf.squeeze(indata)
    return indata


class RedisManager:
    """ REDIS Manager Class for homework2 ML4IoT """
    def __init__(self, _hostname: str = REDIS_HOST, _port: int = REDIS_PORT, _password: str = REDIS_PASSWORD):
        self.client = None
        try:
            self.client = redis.Redis(host=_hostname, port=_port, password=_password)
            is_connected = self.client.ping()
            print('Redis client is connected ', is_connected)
        except Exception as ex:
            print(ex)
        if self.client is None:
            exit(1)
        self.address = ''

    def init_timeseries(self, address):
        self.address = address
        self.client.delete(f'{address}:battery')
        self.client.delete(f'{address}:power')
        self.client.delete(f'{address}:plugged')

        # find battery timeseries retention time by checking the memory occupied by a single value.
        battery_memory_usage = 16
        battery_retention = floor(PLAIN_MEMORY_LIMIT / (battery_memory_usage * 0.10)) * 1000
        print(
            f'The memory used to store one value of the battery level timeseries is {battery_memory_usage}, the retention time: {str(battery_retention)} msec.')

        # find power timeseries retention time by checking the memory occupied by a single value.
        power_memory_usage = 16
        power_retention = floor(PLAIN_MEMORY_LIMIT / (power_memory_usage * 0.10)) * 1000
        print(
            f'The memory used to store one value of the power level timeseries is {power_memory_usage}, the retention time: {str(power_retention)} msec.')

        # find plugged timeseries retention time by checking the memory occupied by a single value.
        plugged_memory_usage = 16
        plugged_retention = floor(AGGREGATED_MEMORY_LIMIT / (plugged_memory_usage * 0.10)) * 86400 * 1000
        print(
            f'The memory used to store one value of the power level timeseries is {plugged_memory_usage}, the retention time: {str(plugged_retention)} msec.')

        self.client.ts().create(f'{address}:battery', retention_msecs=battery_retention)
        self.client.ts().create(f'{address}:power', retention_msecs=power_retention)
        self.client.ts().create(f'{address}:plugged', retention_msec=plugged_retention)

        t = self.client.ts().info(f'{address}:power')

        self.client.ts().createrule(source_key=f'{address}:power', dest_key=f'{address}:plugged',
                                    aggregation_type='sum',
                                    bucket_size_msec=86400)

    def check_memory_usage(self):
        battery_memory_usage = self.client.memory_usage(f'{self.address}:battery')
        power_memory_usage = self.client.memory_usage(f'{self.address}:power')
        plugged_memory_usage = self.client.memory_usage(f'{self.address}:plugged')

        print(f'Memory usage: ')
        print(f'{self.address}:battery -> {battery_memory_usage}')
        print(f'{self.address}:power -> {power_memory_usage}')
        print(f'{self.address}:plugged -> {plugged_memory_usage}')

        if battery_memory_usage >= PLAIN_MEMORY_LIMIT:
            print('Battery timeseries violated memory limit.')
        if power_memory_usage >= PLAIN_MEMORY_LIMIT:
            print('Power timeseries violated memory limit.')
        if plugged_memory_usage >= AGGREGATED_MEMORY_LIMIT:
            print('Plugged timeseries violated memory limit')

    def save_data(self, battery, power, tmp):
        self.client.ts().add(f'{self.address}:battery', int(tmp), battery)
        self.client.ts().add(f'{self.address}:power', int(tmp), power)
        return


class TFLiteInterpreter:
    def __init__(self, model_path: str = None):
        if model_path is not None:
            with zipfile.ZipFile(model_path, 'r') as zip_ref:
                zip_ref.extractall("./")
            model_path = os.path.join('./work/', f'{MODEL_NAME}.tflite')
            self.interpreter = tf.lite.Interpreter(model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            self.downsampling_rate = PREPROCESSING_ARGS['downsampling_rate']
            self.sampling_rate_int64 = tf.cast(self.downsampling_rate, tf.int64)
            self.frame_length = int(self.downsampling_rate * PREPROCESSING_ARGS['frame_lenght_in_s'])
            self.frame_step = int(self.downsampling_rate * PREPROCESSING_ARGS['frame_step_in_s'])
            self.num_spectrogram_bins = self.frame_length // 2 + 1
            self.num_mel_bins = PREPROCESSING_ARGS['num_mel_bins']
            self.lower_frequency = PREPROCESSING_ARGS['lower_f']
            self.upper_frequency = PREPROCESSING_ARGS['upper_f']
            self.num_coefficients = PREPROCESSING_ARGS['num_coefficients']

            self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                num_mel_bins=self.num_mel_bins,
                num_spectrogram_bins=self.num_spectrogram_bins,
                sample_rate=self.downsampling_rate,
                lower_edge_hertz=self.lower_frequency,
                upper_edge_hertz=self.upper_frequency
            )

    def predict(self, mfccs):
        self.interpreter.set_tensor(self.input_details[0]['index'], mfccs)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        top_index = np.argmax(output[0])
        probability = output[0, top_index]
        predicted_label = LABELS[top_index]
        print('End prediction with result: ', predicted_label, probability)
        return predicted_label, probability

    def preprocess(self, indata: np.ndarray, sampling_rate):
        audio = tf.convert_to_tensor(indata)
        audio = tf.squeeze(audio)
        sampling_rate_int64 = tf.cast(self.downsampling_rate, tf.int64)
        zero_padding = tf.zeros(sampling_rate - tf.shape(audio), dtype=tf.float32)
        audio_padded = tf.concat([audio, zero_padding], axis=0)
        stft = tf.signal.stft(
            audio_padded,
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            fft_length=self.frame_length
        )
        spectrogram = tf.abs(stft)

        mel_spectrogram = tf.matmul(spectrogram, self.linear_to_mel_weight_matrix)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :self.num_coefficients]
        mfccs = tf.expand_dims(mfccs, 0)  # batch axis
        mfccs = tf.expand_dims(mfccs, -1)  # channel axis
        mfccs = tf.image.resize(mfccs, [32, 32])
        return mfccs


def is_silence(indata, downsampling_rate, frame_length_in_s, db_fs_thresh, duration_time):
    indata = get_audio_from_numpy(indata)
    sampling_rate_float32 = tf.cast(downsampling_rate, tf.float32)
    frame_length = int(frame_length_in_s * sampling_rate_float32)
    stft = tf.signal.stft(
        indata,
        frame_length=frame_length,
        frame_step=frame_length,
        fft_length=frame_length
    )
    spectrogram = tf.abs(stft)
    dbFS = 20 * tf.math.log(spectrogram + 1.e-6)
    energy = tf.math.reduce_mean(dbFS, axis=1)
    non_silence = energy > db_fs_thresh
    non_silence_frames = tf.math.reduce_sum(tf.cast(non_silence, tf.float32))
    non_silence_duration = (non_silence_frames + 1) * frame_length_in_s

    if non_silence_duration > duration_time:
        return False
    else:
        return True


def predict_keyword(indata):
    mfccs = interpreter.preprocess(indata=indata, sampling_rate=sampling_rate)
    label, probability = interpreter.predict(mfccs)
    return label, probability


def sound_check(indata, frames, callback_time, status):
    silence = is_silence(indata, args.samplerate, args.framelength, args.dbt, args.durationtime)
    if silence is False:
        prediction = predict_keyword(indata)
        if prediction[1] > 0.95 and prediction[0] in ['go', 'stop']:
            global keyword
            keyword = prediction[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True,
                                     description='The tflite model used as predictor, must be placed in the same folder of the script and should be named "model17.tflite". Example launch command "python lab1_ex2.py --hostname xxxx --port xxxxx --password xxxx --user xxxx --device x.')
    parser.add_argument('--hostname', nargs='?', help='Redis database hostname')
    parser.add_argument('--port', nargs='?', help='Redis database port')
    parser.add_argument('--password', nargs='?', help='Redis database password')
    parser.add_argument('--user', nargs='?', help='Redis Cloud username')
    parser.add_argument('--resolution', type=str, default='float32')
    parser.add_argument('--samplerate', type=float, default=16000)
    parser.add_argument('--channels', type=int, default=1)
    parser.add_argument('--duration', type=float, default=1)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--framelength', type=float, default=0.004)
    parser.add_argument('--dbt', type=int, default=-267)
    parser.add_argument('--durationtime', type=float, default=0.04)

    args = parser.parse_args()

    # Parse Arguments
    hostname = args.hostname if args.hostname is not None else REDIS_HOST
    password = args.password if args.password is not None else REDIS_PASSWORD
    port = args.port if args.port is not None else REDIS_PORT

    mac_address = hex(uuid.getnode())

    # Initialize REDIS Client
    client = RedisManager(_hostname=hostname, _port=port, _password=password)
    client.init_timeseries(mac_address)

    # Initialize Interpreter
    global interpreter
    interpreter = TFLiteInterpreter(model_path=MODEL_PATH)

    global sampling_rate
    sampling_rate = args.samplerate

    i = 0
    with sd.InputStream(device=args.device, channels=args.channels, samplerate=args.samplerate, dtype=args.resolution,
                        callback=sound_check, blocksize=args.samplerate * args.duration):
        while True:
            if keyword == 'go':
                timestamp = time()
                battery_level = psutil.sensors_battery().percent
                power_plugged = int(psutil.sensors_battery().power_plugged)

                formatted_datetime = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                print(f'{formatted_datetime} {mac_address} : battery : {battery_level}%')
                print(f'{formatted_datetime} {mac_address} : power = {power_plugged}')
                client.save_data(battery=battery_level, power=power_plugged, tmp=datetime.now().timestamp())
            else:
                print('NO DATA SAVING')
            sleep(1)
            i += 1
