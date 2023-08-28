import numpy as np
from sklearn.decomposition import FastICA
from scipy.signal import butter, lfilter


# Функция вычисления ICA-сигнала
def ica(BGR_signal, fps,order):
    
    # Количество кадров в исходных данных
    num_frames = len(BGR_signal)
    
    # Проверка допустимости размера исходных данных
    if(num_frames==0):
        # Массив исходных данных пуст
        raise NameError('EmptyData')
    
    # Проверка допустимости значения fps
    if(fps<9):
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
        R_norm[i] = (R[i] - R.mean())/R.std()
        G_norm[i] = (G[i] - G.mean())/G.std()
        B_norm[i] = (B[i] - B.mean())/B.std()
    

    # Полосовая фильтрация каналов
    R_filtered = bandpass_filter(R_norm, fps, lowcut=0.7, highcut=4.0, order=order)
    G_filtered = bandpass_filter(G_norm, fps, lowcut=0.7, highcut=4.0, order=order)
    B_filtered = bandpass_filter(B_norm, fps, lowcut=0.7, highcut=4.0, order=order)

    Data = np.array([R_filtered,G_filtered,B_filtered]).T

    # print('Data', np.array(Data).shape)
    # pdb.set_trace()

    # Вычисление ICA-сигнала
    # 计算ICA信号
    ica = FastICA(n_components=1, random_state=0)
    signal_ICA = ica.fit_transform(Data).reshape(1, -1)[0]
    # print('signal_ICA',signal_ICA)

    return signal_ICA


def project_ica(BGR_signal, fps, order):
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
    R_filtered = bandpass_filter(R_norm, fps, lowcut=0.7, highcut=4.0, order=order)
    G_filtered = bandpass_filter(G_norm, fps, lowcut=0.7, highcut=4.0, order=order)
    B_filtered = bandpass_filter(B_norm, fps, lowcut=0.7, highcut=4.0, order=order)

    Data = np.array([R_filtered, G_filtered, B_filtered]).T

    # print('Data', np.array(Data).shape)
    # pdb.set_trace()

    # Вычисление ICA-сигнала
    # 计算ICA信号
    ica = FastICA(n_components=2, random_state=0)

    # Матрица коэффициентов
    # projection_matrix = np.array([[-0.4082, -0.4082, 0.8165],  [0.7071, -0.7071, 0]])   # refer to： Project-ICA
    projection_matrix = np.array([[0, 1, -1], [-2, 1, 1]])
    S = np.matmul(projection_matrix, Data.T)

    # Полосовая фильтрация (здесь S[0,:] и S[1,:] подобны Xs и Ys в методе CHROM)
    S0 = bandpass_filter(S[0], fps, lowcut=0.7, highcut=4.0, order=order)
    S1 = bandpass_filter(S[1], fps, lowcut=0.7, highcut=4.0, order=order)

    # Здесь S[0,:] - S1, S[1,:] - S2 по алгоритму в Wang2017_2
    std = np.array([1, np.std(S0) / np.std(S1)])
    h = np.matmul(std, S)

    signal_ICA = ica.fit_transform(S.T).T

    best_cor = 0
    idx = 0
    for i in range(signal_ICA.shape[0]):
        cor = np.abs(np.corrcoef(signal_ICA[i], h))[0,1]
        if cor > best_cor:
            idx = i
            best_cor = cor
    # print('signal_ICA[idx]',signal_ICA[idx].shape)
    # pdb.set_trace()
    return signal_ICA[idx]