import argparse
import json
import os
import zipfile
from datetime import datetime
from time import sleep, time

import numpy as np
import paho.mqtt.client as mqtt
import sounddevice as sd
import tensorflow as tf
import tqdm

global LABELS  # = ['music', 'speech',]
LABELS = ['music', 'speech']

PREPROCESSING_ARGS = {
    'downsampling_rate': 16000,
    'frame_lenght_in_s': 0.032,
    'frame_step_in_s': 0.032,
    'num_mel_bins': 20,
    'lower_f': 100,
    'upper_f': 8000,
    'num_coefficients': 20
}

MODEL_PATH = 'model.tflite.zip'
MODEL_NAME = MODEL_PATH.split('/')[-1].split('.')[0]

is_silence_flag: bool = True
keyword: str = 'music'
prediction_probability: float = 0.0


def get_audio_from_numpy(indata):
    indata = tf.convert_to_tensor(indata, dtype=tf.float32)
    indata = 2 * ((indata + 32768) / (32767 + 32768)) - 1
    indata = tf.squeeze(indata)
    return indata


class TFLiteInterpreter:
    def __init__(self, model_path: str = None):
        if model_path is not None:
            with zipfile.ZipFile(model_path, 'r') as zip_ref:
                zip_ref.extractall("./")
            print(MODEL_NAME)
            model_path = os.path.join('./model.tflite/tflite_models/', f'{MODEL_NAME}.tflite')
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
        # print('End prediction with result: ', predicted_label, probability)
        return predicted_label, probability

    def preprocess(self, indata: np.ndarray, sampling_rate):
        indata = get_audio_from_numpy(indata)
        audio = tf.convert_to_tensor(indata)
        audio = tf.squeeze(audio)
        audio_padded = audio
        audio_padded = tf.cast(audio_padded, tf.float32)
        sampling_rate_int64 = tf.cast(self.downsampling_rate, tf.int64)
        # zero_padding = tf.zeros(sampling_rate - tf.shape(audio), dtype=tf.float32)
        # audio_padded = tf.concat([audio, zero_padding], axis=0)
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


class MqttClient:
    def __init__(self, broker: str, port: int, keepalive: int = 60):
        self.broker = broker
        self.port = port
        self.keepalive = keepalive
        self.client = mqtt.Client()

    def connect(self):
        self.client.connect(host=self.broker, port=self.port, keepalive=self.keepalive)

    def publish(self, topic, message):
        self.client.publish(topic, message)


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
    mfccs = interpreter.preprocess(indata=indata, sampling_rate=int(sampling_rate))
    label, probability = interpreter.predict(mfccs)
    global keyword
    keyword = label
    global prediction_probability
    prediction_probability = probability
    return label, probability


def sound_check(indata, frames, callback_time, status):
    global is_silence_flag
    is_silence_flag = is_silence(indata, args.samplerate, args.framelength, args.dbt, args.durationtime)
    if is_silence_flag is False:
        prediction = predict_keyword(indata)
        global keyword
        global prediction_probability
        if prediction[1] > 0.6 and prediction[0] in ['music', 'speech', ]:

            # global keyword
            keyword = prediction[0]
            # global prediction_probability
            prediction_probability = prediction[1]
        else:
            keyword = f'probably {prediction[0]}'
            prediction_probability = prediction[1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True,
                                     description='The tflite model used as predictor, must be placed in the same folder of the script and should be named "model17.tflite". Example launch command "python lab1_ex2.py --hostname xxxx --port xxxxx --password xxxx --user xxxx --device x.')
    parser.add_argument('--resolution', type=str, default='int16')
    parser.add_argument('--samplerate', type=float, default=16000)
    parser.add_argument('--channels', type=int, default=1)
    parser.add_argument('--duration', type=float, default=0.3)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--framelength', type=float, default=0.004)
    parser.add_argument('--dbt', type=int, default=-266)
    parser.add_argument('--durationtime', type=float, default=0.04)

    args = parser.parse_args()

    broker = 'hermesdmpmqttbroker.e3ebhhb8bkhhd6fh.westeurope.azurecontainer.io'
    port = 1883
    topic = 's290453/sensor_data/s1'


    def on_connect(client, userdata, flags, rc):
        print("Connected with result code " + str(rc))


    client = mqtt.Client()
    client.on_connect_fail = on_connect
    client.connect(broker, port, 60)

    # Initialize Interpreter

    global interpreter
    interpreter = TFLiteInterpreter(model_path=MODEL_PATH)

    global sampling_rate
    sampling_rate = args.samplerate

    with sd.InputStream(device=args.device, channels=args.channels, samplerate=args.samplerate, dtype=args.resolution,
                        callback=sound_check, blocksize=int(args.samplerate * args.duration)):
        while True:
            if not is_silence_flag:
                timestamp = time()
                formatted_datetime = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                data = {
                    "listening_to": keyword,
                    'probability': int(prediction_probability * 100),
                    'timestamp': formatted_datetime
                }
                print("publishing", data)
                client.publish(topic, json.dumps(data))
            else:
                timestamp = time()
                formatted_datetime = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                data = {
                    "listening_to": 'silence',
                    'probability': 0,
                    'timestamp': formatted_datetime
                }
                print("publishing", data)
                client.publish(topic, json.dumps(data))
            # sleep(args.duration)
            for _ in tqdm.tqdm(range(int(args.duration * 1000))):
                sleep(0.9 / 1000)
