from scipy import fft, arange
import numpy as np
import matplotlib.pyplot as plt
import pyaudio
from scipy.io import wavfile
import os
import csv
import wave
import wavio
import noisereduce as nr
import math as math
import datetime
import calendar
import time
import sys
import signal
from concurrent.futures import thread

TIMESTAMP = str(calendar.timegm(time.gmtime()))
CHUNK = 4410
RATE = 44100
CHANNELS = 1
OUTPUT_WAV_FILE = './streams/stream_' + TIMESTAMP + '.wav'
OUTPUT_PEAK_FILE = './peaks/peaks_' + TIMESTAMP + '.csv'
NOISE_FILE = './noise.wav'
SLOPE = 10  # Used to control the peak detection sensitivity

global AUDIO

total_list = [210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 270.0, 280.0, 290.0, 300.0,
              310.0, 320.0, 330.0, 340.0, 350.0, 360.0, 370.0, 380.0, 390.0, 400.0,
              410.0, 420.0, 430.0, 440.0, 450.0, 460.0, 470.0, 480.0, 490.0, 500.0,
              510.0, 520.0, 530.0, 540.0, 550.0, 560.0, 570.0, 580.0, 590.0, 600.0,
              610.0, 620.0, 630.0, 640.0, 650.0, 660.0, 670.0, 680.0, 690.0, 700.0,
              710.0, 720.0, 730.0, 740.0, 750.0, 760.0, 770.0, 780.0, 790.0, 800.0,
              810.0, 820.0, 830.0, 840.0, 850.0, 860.0, 870.0, 880.0, 890.0, 900.0,
              910.0, 920.0, 930.0, 940.0, 950.0, 960.0, 970.0, 980.0, 990.0, 1000.0, 'classification']

# noise sample file
noise_rate, noise_data = wavfile.read(NOISE_FILE)
noise_data = noise_data / 1.0

p = pyaudio.PyAudio()

AUDIO = []


def signal_handler(sig, frame):
    print('\nYou finished the recording...\n')
    write_audio_file()


signal.signal(signal.SIGINT, signal_handler)


def frequency_sepectrum(x, sf):
    """
    Derive frequency spectrum of a signal from time domain
    :param x: signal in the time domain
    :param sf: sampling frequency
    :returns frequencies and their content distribution
    """
    x = x - np.average(x)  # zero-centering
    n = len(x)
    k = np.arange(n)
    tarr = n / float(sf)
    frqarr = k / float(tarr)  # two sides frequency range
    frqarr = frqarr[range(n // 2)]  # one side frequency range
    x = fft.fft(x) / n  # fft computing and normalization
    x = x[range(n // 2)]

    return frqarr, abs(x)


def calc_slope(p_x, p_y, c_x, c_y, n_x, n_y):
    return max(math.degrees(math.atan((c_x-p_x)/(c_y-p_y))), math.degrees(math.atan((n_x-c_x)/(c_y-n_y))))


def check_peak(p_x, p_y, c_x, c_y, n_x, n_y):
    if p_x < c_x and n_x < c_x and calc_slope(p_x, p_y, c_x, c_y, n_x, n_y) > SLOPE:
        return True
    else:
        return False


def write_to_file(writer, list):
    writer.writerow(list)


def process_second(second):
    with open(OUTPUT_PEAK_FILE, 'a') as f:
        writer = csv.writer(f, delimiter=',')
        for i in range(10):
            peak_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, '-']

            data = second[i*CHUNK:(i+1)*CHUNK]
            t = np.arange(len(data)) / float(RATE)

            frq, X = frequency_sepectrum(data, RATE)

            for j in enumerate(zip(frq, X)):
                if j[0] > 0:
                    if 210 <= frq[j[0]] <= 1000 and check_peak(X[j[0]-1], frq[j[0]-1], X[j[0]], frq[j[0]], X[j[0]+1], frq[j[0]+1]):
                        index = total_list.index(frq[j[0]])
                        peak_list[index] = 1
            write_to_file(writer, peak_list)


def write_audio_file():
    global AUDIO
    print('\nwriting audio file ' + OUTPUT_WAV_FILE)
    AUDIO = np.array(AUDIO, dtype=np.int16)
    p.terminate()

    wf = wave.open(OUTPUT_WAV_FILE, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(AUDIO))
    wf.close()

    sys.exit(0)


def noise_reduce(second):
    # global AUDIO
    reduced_second = nr.reduce_noise(
        second.flatten(), noise_data.flatten(), verbose=False)
    process_second(reduced_second)


stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True,
                input_device_index=6,
                frames_per_buffer=CHUNK)

#
# Collect 1 sec
#XCVV        i = 0
