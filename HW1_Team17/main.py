from numpy import dtype
import sounddevice as sd
import argparse as ap
from scipy.io.wavfile import write
import tensorflow as tf
from time import time
import os
import re


parser = ap.ArgumentParser()
parser.add_argument('--resolution', type=str, default='int16')
parser.add_argument('--samplerate', type=int, default=16000)
parser.add_argument('--channels', type=int, default=1)
parser.add_argument('--duration', type=int, default=1)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--framelength', type=float, default=0.004)
parser.add_argument('--dbt', type=int, default=-133)
parser.add_argument('--durationtime', type=float, default=0.04)
# parser.add_argument(...)
args = parser.parse_args()
# args.resolution will store the value of --resolution

def get_audio_from_numpy(indata):
    indata = tf.convert_to_tensor(indata, dtype=tf.float32)
    indata = 2 * ((indata + 32768) / (32767 + 32768)) - 1
    indata = tf.squeeze(indata)
    return indata


def is_silence(indata, downsampling_rate, frame_length_in_s, dbFSthresh, duration_time):
    #audio, sampling_rate, label = get_audio_and_label(filename)
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
    non_silence = energy > dbFSthresh
    non_silence_frames = tf.math.reduce_sum(tf.cast(non_silence, tf.float32))
    non_silence_duration = (non_silence_frames + 1) * frame_length_in_s

    if non_silence_duration > duration_time:
        return False
    else:
        return True



#callback should define if is it silence or not
def callback(indata, frames, callback_time, status):
    global verbose
    global silence
    silence = is_silence(indata, args.samplerate, args.framelength, args.dbt, args.durationtime)
    if silence is False:
        timestamp = time()
        write(f'{timestamp}.wav', args.samplerate, indata)
        size_in_bytes = os.path.getsize(f'{timestamp}.wav')
        size_in_kb = size_in_bytes // 1024
        res_byte = int(re.findall(r'\d+', args.resolution)[0])
        print(res_byte)
        est_size_bytes = args.samplerate * args.duration * (res_byte // 8) * args.channels
        est_size_kb = est_size_bytes // 1024
        if verbose is True:
            print(f'Estimated size: {est_size_kb}')
            print(f'Size {size_in_kb} KB')







print('Start recording')

verbose = True
print('Verbose mode:', verbose)
#print('Audio storage:', store_audio)
with sd.InputStream(device=args.device, channels=args.channels, samplerate=args.samplerate, dtype=args.resolution, callback=callback, blocksize=args.samplerate * args.duration):
    while True:
        key = input()
        if key in ['Q', 'q']:
            print('Stop recording')
            break

        if key in ['V', 'v']:
            verbose = not verbose
            print('Verbose mode:', verbose)



# optimal parameters:
# dbthresh : -80
# durationtime : 0.1
# frame_length : 0.04