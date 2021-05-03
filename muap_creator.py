# Программа для выделения отдельных движений
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
nums = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15"]
folder = "male0"


def muap_creator(nums, folder):
    for i in nums:
        x = np.loadtxt("data_pure/{}/0{}.txt".format(folder, i))

        # Центрирование
        x -= np.mean(x)

        # Подавим сигнал в полосе от 0 до 5Гц. Для этого расчитаем фильтр
        SPS = 1000.0
        hflt = signal.firls(513, [0., 5., 7., SPS / 2], [0., 0., 1.0, 1.0], fs=SPS)
        w, h = signal.freqz(hflt, fs=SPS)
        x = np.convolve(hflt, x, 'same')
        # plt.figure(figsize=(10, 6))
        # plt.plot(x)
        # plt.figure(1)

        # Локализация MUAP для одного движения, разбиение сигнала на векторы одинаковой длины,
        # содержащие только ЭМГ сигнал для одного движения
        def emg_muap(x):
            N = len(x)
            wnd_size = 600
            muap_left = 600
            muap_right = 600
            muaps = np.zeros((muap_left + muap_right, 500))
            numMuaps = 0
            for k in range(0, N):
                # берем кусочек сигнала в текущей позиции окна
                xWnd = np.abs(x[k:k + wnd_size])
                xWnd = xWnd - np.mean(xWnd)
                maxValue = np.max(xWnd)
                maxIndices = np.argmax(xWnd)
                if (maxIndices == wnd_size // 2) and (maxValue > 3000):
                    a = k + wnd_size // 2 - muap_left
                    b = k + wnd_size // 2 + muap_right
                    if a < 0 or b > N:
                        continue
                    muaps[:, numMuaps] = x[a:b]
                    numMuaps = numMuaps + 1
            muaps = muaps[:, :numMuaps]
            return (muaps)


        muaps = emg_muap(x)


        # saving muaps for each type of movements
        # dir = "dataset/{}/0{}".format(folder, i)
        # os.makedirs(dir)
        #
        # for j in range(0, muaps.shape[1]):
        #     np.savetxt(dir + "/{}.txt".format(j), muaps[:, j])


        #
        category = 0
        if i == "00":
            category = 0
        if i == "13" or i == "01" or i == "04" or i == "07" or i == "10":
        # if i == "13":
            category = 1
        # if i == "14":
        if i == "14" or i == "02" or i == "05" or i == "08" or i == "11":
            category = 2
        if i == "15" or i == "03" or i == "06" or i == "09" or i == "12":
        # if i == "15":
            category = 3
        if folder == "male22" or folder == "male23":
            for j in range(0, muaps.shape[1]):
                np.savetxt("full_dataset_test/{}/{}_{}_{}.txt".format(category, folder, i, j), muaps[:, j])
                print("full_dataset_test/{}/{}_{}_{}.txt".format(category, folder, i, j))
            continue
        for j in range(0, muaps.shape[1]):
            np.savetxt("full_dataset/{}/{}_{}_{}.txt".format(category, folder, i, j), muaps[:, j])
            print("full_dataset/{}/{}_{}_{}.txt".format(category, folder, i, j))



        # plt.figure(figsize=(10, 6))
        # plt.plot(muaps[:, 0:muaps.shape[1]])
        # plt.figure(2)
        # plt.grid(True)
        # man = plt.get_current_fig_manager()
        # man.canvas.set_window_title("{} moves".format(muaps.shape[1]))
        #
        # plt.show()


# muap_creator(nums, folder)
# nums = ["00", "13", "14", "15"]
folders = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23"]
for f in folders:
    muap_creator(nums, "male{}".format(f))

