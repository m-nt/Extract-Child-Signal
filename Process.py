from scipy import io, signal
import matplotlib.pyplot as plt
import numpy as np
import heartpy as hp
import padasip as pa


def ReShape(data, shape):
    data_shaped = np.zeros(shape)
    for i in range(len(data_shaped[0, 0:])):
        for j in range(len(data_shaped[0:, 0])):
            data_shaped[j, i] = data[i, j]
    return data_shaped


def PreProcess(meandata, index, start, end, raw_data=False):
    data = meandata[index, start: end]
    if raw_data:
        x = np.arange(len(data))
        return (x, data)
    Base_drift = hp.remove_baseline_wander(data, 1000)
    Base_drift_X = np.arange(len(Base_drift))

    sos = signal.butter(2, 0.1, 'low', output='sos')
    filtered = signal.sosfilt(sos, Base_drift)

    filtered = hp.smooth_signal(filtered, 1000, 57, 3)
    mean = np.mean(filtered)
    if mean > 0:
        filtered = filtered - abs(mean)
    else:
        filtered = filtered + abs(mean)

    return (Base_drift_X, filtered)


# dict_keys(['__header__', '__version__', '__globals__', 'val'])
f = io.loadmat("r01_edfm.mat")

basedata = f["val"]

x, andomen_raw = PreProcess(basedata, 3, 4000, 8000, raw_data=True)
Ax_1, Abdomen_1 = PreProcess(basedata, 1, 4000, 8000)
Ax_2, Abdomen_2 = PreProcess(basedata, 2, 4000, 8000)
Ax_3, Abdomen_3 = PreProcess(basedata, 3, 4000, 8000)
Ax_4, Abdomen_4 = PreProcess(basedata, 4, 4000, 8000)

AbdomensHold = np.array([Abdomen_1, Abdomen_2, Abdomen_3, Abdomen_4])
Abdomens = ReShape(AbdomensHold, (4000, 4))

Dx, Direction = PreProcess(basedata, 0, 4000, 8000)

Abdomens_number = 3
f = pa.filters.FilterNLMS(n=4, mu=0.0001, w="random")
y, e, w = f.run(Direction, Abdomens)

plt.figure(figsize=(10, 5))
plt.subplot(221)
plt.title("Pre-Processed")
plt.plot(Abdomens[:, Abdomens_number], "b", label="Abdomen")
plt.legend()
plt.subplot(222)
plt.title("raw data")
plt.plot(andomen_raw, "y", label="raw signal")
plt.legend()
plt.tight_layout()
plt.subplot(223)
plt.title("Filter error (FECG Extraction)")
plt.plot(e, "r", label="fetal signal")
plt.legend()
plt.tight_layout()
plt.subplot(224)
plt.title("Pre-Processed")
plt.plot(Direction, "g", label="Direction")
plt.legend()
plt.tight_layout()
plt.show()
