from scipy import fft, arange
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
import csv


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


def calc_slope(x_1, Y_1, x_2, y_2):
    return (y_2 - Y_1) / (x_2 - x_1)


def check_peak(p_x, p_y, c_x, c_y, n_x, n_y):
    if p_x < c_x and n_x < c_x and max(calc_slope(p_x, p_y, c_x, c_y),
                                       abs(calc_slope(c_x, c_y, n_x, n_y))) < 20:
        return True
    else:
        return False


def write_to_file(write, list):
    writer.writerow(list)


total_list = [210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 270.0, 280.0, 290.0, 300.0,
              310.0, 320.0, 330.0, 340.0, 350.0, 360.0, 370.0, 380.0, 390.0, 400.0,
              410.0, 420.0, 430.0, 440.0, 450.0, 460.0, 470.0, 480.0, 490.0, 500.0,
              510.0, 520.0, 530.0, 540.0, 550.0, 560.0, 570.0, 580.0, 590.0, 600.0,
              610.0, 620.0, 630.0, 640.0, 650.0, 660.0, 670.0, 680.0, 690.0, 700.0,
              710.0, 720.0, 730.0, 740.0, 750.0, 760.0, 770.0, 780.0, 790.0, 800.0,
              810.0, 820.0, 830.0, 840.0, 850.0, 860.0, 870.0, 880.0, 890.0, 900.0,
              910.0, 920.0, 930.0, 940.0, 950.0, 960.0, 970.0, 980.0, 990.0, 1000.0, 'classification']

# file 1
with open("./combo.csv", "a") as f:
    writer = csv.writer(f, delimiter=',')

    here_path = os.path.dirname(os.path.realpath(__file__))
    wav_file_name = '2020-05-14-19:22:45.wav'
    wave_file_path = os.path.join(here_path, wav_file_name)
    sr, signal = wavfile.read(wave_file_path)
    length = signal.shape[0] / sr
    i = 0
    while i <= 44:
        peak_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, '-']
        if i >= 0:
            y = signal[i*4410:(i+1) * 4410, 0]
            t = np.arange(len(y)) / float(sr)

            frq, X = frequency_sepectrum(y, sr)
            for j in enumerate(zip(frq, X)):
                if j[0] > 0:
                    if 210 <= frq[j[0]] <= 1000 and check_peak(X[j[0]-1], frq[j[0]-1], X[j[0]], frq[j[0]], X[j[0]+1], frq[j[0]+1]):
                        index = total_list.index(frq[j[0]])
                        peak_list[index] = 1
            index = total_list.index('classification')
            # not decisive
            if not 20 <= i <= 30 or 50 <= i <= 60 or 70 <= i <= 100:
                # for sure bee vs for sure no bee
                if 30 < i < 50 or 60 < i < 70:
                    peak_list[index] = 'red-bee'
                else:
                    peak_list[index] = 'no-bee'
                write_to_file(writer, peak_list)
        i = i + 1
f.close()
