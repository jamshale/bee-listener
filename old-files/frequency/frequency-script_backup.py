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
        # slope_list.append(max(calc_slope(p_x, p_y, c_x, c_y),
        #                       abs(calc_slope(c_x, c_y, n_x, n_y))))
        return True
    else:
        return False


def write_to_file(write, list):
    writer.writerow(list)


with open("./2020-05-14-19:19:22_processed.csv", "a") as f:
    writer = csv.writer(f, delimiter=',')

    total_list = []
    # total_list = [22.05, 33.075, 44.1, 55.125, 66.15, 77.175, 88.2, 99.225, 110.25, 121.275, 132.3, 143.325, 154.35, 165.375, 176.4, 187.425, 198.45, 209.475, 220.5, 231.525, 242.55, 253.57500000000002, 264.6, 275.625, 286.65, 297.675, 308.7, 319.725, 330.75, 341.775, 352.8, 363.825, 374.85, 385.875,
    #               396.9, 407.925, 418.95, 429.975, 441.0, 452.025, 463.05, 474.075, 485.1, 496.125, 507.15000000000003, 518.175, 529.2, 562.275, 584.325, 595.35, 606.375, 628.425, 716.625, 727.65, 738.675, 749.7, 760.725, 848.925, 870.975, 882.0, 893.025, 904.05, 915.075, 926.1, 937.125, 948.15, 970.2, 981.225, 992.25]
    here_path = os.path.dirname(os.path.realpath(__file__))
    wav_file_name = '../bee-sounds/2020-05-14-19:19:22.wav'
    wave_file_path = os.path.join(here_path, wav_file_name)
    sr, signal = wavfile.read(wave_file_path)
    length = signal.shape[0] / sr
    # use the first channel (or take their average, alternatively)
    i = 0
    while i <= 820:
        peak_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        if i >= 0:
            y = signal[i*4410:(i+1) * 4410, 0]
            t = np.arange(len(y)) / float(sr)

            plt.figure()
            plt.subplot(2, 1, 1)
            plt.plot(t, y)
            plt.xlabel('t')
            plt.ylabel('y')

            frq, X = frequency_sepectrum(y, sr)

            # process known peaks
            # peak_list = []
            # slope_list = []
            # for j in enumerate(zip(frq, X)):
            #     if j[0] > 0:
            #         if 20 <= frq[j[0]] <= 1000 and check_peak(X[j[0]-1], frq[j[0]-1], X[j[0]], frq[j[0]], X[j[0]+1], frq[j[0]+1]):
            #             if frq[j[0]] not in total_list:
            #                 total_list.append(frq[j[0]])
            # index = total_list.index(frq[j[0]])
            # peak_list[index] = 1
            # print(peak_list)
            # print(" ")
            # print(slope_list)
            # print("\n\n")

            # write_to_file(writer, peak_list)

            plt.subplot(2, 1, 2)
            plt.xlim(right=1000)
            plt.xlim(left=20)
            plt.plot(frq, X, 'b')
            left, right = plt.xlim()
            plt.xlabel('Freq (Hz)')
            plt.ylabel('|X(freq)|')
            plt.tight_layout()
            plt.show()
        i = i + 1
    # total_list.sort()
    # write_to_file(writer, total_list)
