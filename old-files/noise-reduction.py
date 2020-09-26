import noisereduce as nr
from scipy.io import wavfile
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import librosa
import wavio

CHUNK = 4410
RATE = 44100
CHANNELS = 1
OUTPUT_FILE = "clean_1595797351.wav"

rate, data = wavfile.read('./streams/stream_1595797351.wav')
n_rate, n_data = wavfile.read('./noise.wav')
data = data / 32768.0
n_data = n_data / 32768.0
# print(data[0])
# noise = data[100:300]
reduce_noise = nr.reduce_noise(
    audio_clip=data.flatten(), noise_clip=n_data.flatten(), verbose=True)

wavio.write(OUTPUT_FILE, reduce_noise, RATE, sampwidth=2)
