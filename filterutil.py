import numpy as np


def QRS_Extraction(data, thrsh, drange):
    ECG_length = len(data) + 1
    QRS_ECG = np.zeros((ECG_length+2*drange), float)
    ECG = data
    ECG = np.append(ECG, 0)

    for i in range(drange, ECG_length):
        if abs(ECG[i]) > abs(ECG[i-1]) and abs(ECG[i]) > abs(ECG[i+1]) and abs(ECG[i]) > thrsh:
            QRS_ECG[i] = ECG[i]
            QRS_ECG[i: i-drange+1: -1] = ECG[i: i-drange+1: -1]
            QRS_ECG[i: i+drange-1] = ECG[i: i+drange-1]
    return QRS_ECG[drange+1: ECG_length+drange]
