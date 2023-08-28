import numpy as np
from sklearn.decomposition import PCA
from scipy.signal import butter, lfilter
from signal_filter import *

# Функция вычисления ICA-сигнала
def pca(BGR_signal, fps, order):
    # Количество кадров в исходных данных
    num_frames = len(BGR_signal)

    # Проверка допустимости размера исходных данных
    if (num_frames == 0):
        # Массив исходных данных пуст
        raise NameError('EmptyData')

    # Проверка допустимости значения fps
    if (fps < 9):
        # Недопустимое значение fps, для работы полосового фильтра требуется fps>=9
        raise NameError('WrongFPS')

    # Разделение исходного сигнала на каналы R,G,B (R, G, B)
    R = BGR_signal[:, 2]
    G = BGR_signal[:, 1]
    B = BGR_signal[:, 0]

    # Нормализация
    R_norm = np.zeros(num_frames)
    G_norm = np.zeros(num_frames)
    B_norm = np.zeros(num_frames)

    for i in range(num_frames):
        R_norm[i] = (R[i] - R.mean()) / R.std()
        G_norm[i] = (G[i] - G.mean()) / G.std()
        B_norm[i] = (B[i] - B.mean()) / B.std()

    # Полосовая фильтрация каналов
    R_filtered = bandpass_filter(R_norm, fps, cutoff=(0.7,4.0), order=order)
    G_filtered = bandpass_filter(G_norm, fps, cutoff=(0.7,4.0), order=order)
    B_filtered = bandpass_filter(B_norm, fps, cutoff=(0.7,4.0), order=order)

    Data = np.array([R_filtered, G_filtered, B_filtered]).T

    # print('Data', np.array(Data).shape)
    # pdb.set_trace()

    # Вычисление ICA-сигнала
    pca = PCA(n_components=1, random_state=0)
    signal_PCA = pca.fit_transform(Data).reshape(1, -1)[0]

    return signal_PCA