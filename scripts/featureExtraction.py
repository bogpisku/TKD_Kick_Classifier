import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import glob


def getPeaks(signal, win=3):
    peaksMax, peaksMin = [], []
    if all(signal[0] > x for x in signal[1:win]):
        peaksMax.append(0)
    elif all(signal[0] < x for x in signal[1:win]):
        peaksMin.append(0)

    for i in range(1, win):
        if all(signal[i] > x for x in signal[0:i]) and all(signal[i] > x for x in signal[i + 1:i + 1 + win]):
            peaksMax.append(i)
        elif all(signal[i] < x for x in signal[0:i]) and all(signal[i] < x for x in signal[i + 1:i + 1 + win]):
            peaksMin.append(i)

    for i in range(win, len(signal) - win):
        if all(signal[i] > x for x in signal[i - win:i]) and all(signal[i] > x for x in signal[i + 1:i + 1 + win]):
            peaksMax.append(i)
        elif all(signal[i] < x for x in signal[i - win:i]) and all(signal[i] < x for x in signal[i + 1:i + win]):
            peaksMin.append(i)

    for i in range(len(signal) - win, len(signal) - 1):
        if all(signal[i] > x for x in signal[i - win:i]) and all(signal[i] > x for x in signal[i + 1:len(signal)]):
            peaksMax.append(i)
        elif all(signal[i] < x for x in signal[i - win:i]) and all(signal[i] < x for x in signal[i + 1:len(signal)]):
            peaksMin.append(i)

    if all(signal[-1] > x for x in signal[-1 - win:-2]):
        peaksMax.append(i)
    elif all(signal[-1] < x for x in signal[-1 - win:-2]):
        peaksMin.append(i)

    # return peaksMax, peaksMin
    # print(len(peaksMax) + len(peaksMin))
    return len(peaksMax) + len(peaksMin)


def loadData(path):
    data = np.array(pd.read_csv(path, index_col=False))
    raw = data[:, 0:3]
    return raw


def add_magnitude(data):
    acm = (np.linalg.norm(data, axis=1)).reshape((len(data), 1))
    signals = np.hstack((data, acm))
    return signals


def rms(signal):
    square = 0
    # Calculate square
    for i in range(len(signal)):
        square += (signal[i] ** 2)
        # Calculate Mean
    mean = square / len(signal)
    # Calculate Root
    root = np.sqrt(mean)
    return root


def getFeatures(signals, cl, win=60, ol=0.5):
    window = win
    overlap = int(ol * window)
    features = []
    i = 0
    # feature order: 4 signals ACx, ACy, ACz, ACm
    # 0-3: Mean Values Mx, My, Mz, Mm
    # 4-7: Average of peak freq APFx, APFy, APFz, APFm
    # 8: Variance of APF VarAPF(x,y,z)
    # 9-12: Root Mean Square RMSx, RMSy, RMSz, RMSm
    # 13-16: Std Dev Stdx, Stdy, Stdz, Stdm
    # 17-20: MinMax difference MMx, MMy, MMz, MMm
    # 21-23: Correlation Cxy, Cxz, Cyz
    # 24: Class
    while True:
        start = int(i * overlap)
        end = start + window
        i += 1
        if end > len(signals):
            break
        else:
            # signals, dcm = filterData(data[start:end,:], sF=50, cF = 0.1)
            descriptor = np.zeros((25,))
            # Mean
            for j in range(0, 4):
                descriptor[j] = np.mean(signals[start:end, j])
                # descriptor[j] = np.mean(signals[:,j])
            # APF
            for j in range(4, 8):
                descriptor[j] = getPeaks(signals[start:end, j % 4])
                # descriptor[j] = getPeaks(signals[:,j%4])
            # varAPF
            descriptor[8] = np.var(descriptor[4:7])
            # rms
            for j in range(9, 13):
                descriptor[j] = rms(signals[start:end, j % 9])
                # descriptor[j] = rms(signals[:,j%9])
            # Std
            for j in range(13, 17):
                descriptor[j] = np.std(signals[start:end, j % 13])
                # descriptor[j] = np.std(signals[:,j%13])
            # MinMax
            for j in range(17, 21):
                descriptor[j] = np.max(signals[start:end, j % 17]) - np.min(signals[start:end, j % 17])
                # descriptor[j] = np.max(signals[:,j%17]) - np.min(signals[:,j%17])
            # Correlation between axes
            descriptor[21] = (np.cov(signals[start:end, 0], signals[start:end, 1])[0, 1]) / (
                    descriptor[13] * descriptor[14])
            descriptor[22] = (np.cov(signals[start:end, 0], signals[start:end, 2])[0, 1]) / (
                    descriptor[13] * descriptor[15])
            descriptor[23] = (np.cov(signals[start:end, 1], signals[start:end, 2])[0, 1]) / (
                    descriptor[14] * descriptor[15])
            # descriptor[21] = (np.cov(signals[:, 0], signals[:, 1])[0,1])/(descriptor[13]*descriptor[14])
            # descriptor[22] = (np.cov(signals[:, 0], signals[:, 2])[0,1])/(descriptor[13]*descriptor[15])
            # descriptor[23] = (np.cov(signals[:, 1], signals[:, 2])[0,1])/(descriptor[14]*descriptor[15])
            descriptor[24] = cl
            # corrX2,_ = pearsonr(signals[start:end, 0], signals[start:end, 1])
            # print(corrX1, corrX2)
            features.append(descriptor)

    features = np.array(features)
    return features


def processData(path, cl, win=50, ol=0.5):
    # load data
    data = loadData(path)
    print("Adding magnitude")
    signals = add_magnitude(data)

    # signals, dcm = filterData(data, sF=100, cF = 0.25)
    # windowing data 60 samples
    print("Getting all Features")
    features = getFeatures(signals, cl, win, ol)
    # plt.plot(signals[:, 3])
    # plt.show()
    return features


classes = ["yeop_chagi", "ap_chagi", "pi_chagi","step"]

headers = ["Mx", "My", "Mz", "Mm", "APFx", "APFy", "APFz", "APFm", "varAPF",
           "RMSx", "RMSy", "RMSz", "RMSm", "SDx", "SDy", "SDz", "SDm",
           "MMx", "MMy", "MMz", "MMm", "Cxy", "Cxz", "Cyz", "Class"]

directory = os.getcwd() + "/DataSets/50_kicks/linear_acceleration/"

dataset = []

print("get Files from main directory: "+str(directory))
for i in range(len(classes)):
    files = glob.glob(directory + classes[i] + "/*")
    print("Get files from class directory : "+directory + classes[i] + "/*")
    for file in files:
        print("Opening file"+str(file))
        data = processData(file, i, win=125, ol=0.5)
        print("Append to new File")
        # print(data)
        dataset.append(data)

dataset = np.concatenate(dataset, axis=0)
df = pd.DataFrame(dataset)
print("save new File")
df.to_csv("DataSets/50_kicks/linear_acceleration/full_dataset.csv", header=headers, index=False)
