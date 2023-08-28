import numpy as np
from scipy.signal import butter, lfilter


# Функция вычисления POS-сигнала
def pos(BGR_signal, fps, l, order):
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

    # Если длина нахлеста не задана
    if (l == None):
        # В статье Wang2017_2 использовалась длина равная 20 при fps=20, для сохранения пропорций tw = fps
        l = int(fps)
    elif (l > 0):
        l = l // 1
    else:
        # Недопустимое значение длины нахлеста
        raise NameError('WrongLength')

    # Проверка допустимости размера исходных данных
    if (num_frames < l):
        # Недопустимая длина исходных данных, длина исходных данных должна быть не меньше длины нахлеста
        raise NameError('NotEnoughData')

    # Разделение исходного сигнала на каналы R,G,B
    R = BGR_signal[:, 2]
    G = BGR_signal[:, 1]
    B = BGR_signal[:, 0]


    # Массив данных с которым работает метод (изначально BGR, затем преобразуем в RGB)
    RGB = np.transpose(np.array([R, G, B]))

    H = np.zeros(num_frames)

    for n in range(num_frames - l):
        m = n - l + 1
        # Массив, содержащий часть исходных данных (от m-й до n-й строки)
        C = RGB[m:n, :].T
        if m >= 0:
            # Нормализация
            mean_color = np.mean(C, axis=1)
            diag_mean_color = np.diag(mean_color)
            diag_mean_color_inv = np.linalg.inv(diag_mean_color)
            Cn = np.matmul(diag_mean_color_inv, C)

            # Матрица коэффициентов
            projection_matrix = np.array([[0, 1, -1], [-2, 1, 1]])

            S = np.matmul(projection_matrix, Cn)

            # Полосовая фильтрация (здесь S[0,:] и S[1,:] подобны Xs и Ys в методе CHROM)

            S[0, :] = bandpass_filter(S[0], fps, lowcut=0.7, highcut=4.0, order=order)
            S[1, :] = bandpass_filter(S[1], fps, lowcut=0.7, highcut=4.0, order=order)

            # Здесь S[0,:] - S1, S[1,:] - S2 по алгоритму в Wang2017_2
            std = np.array([1, np.std(S[0, :]) / np.std(S[1, :])])
            h = np.matmul(std, S)

            # Вычисление итогового сигнала
            # Деление на np.std(h) взято с реализации в интернете
            # https://github.com/pavisj/rppg-pos/blob/master/pos_face_seg.py
            # После добавления деления на среднее отклонение всплески данных в результирующем сигнале уходят

            # 计算最终信号
            # np.std(h)除以互联网销售
            # https://github.com/pavisj/rppg-pos/blob/master/pos_face_seg.py
            # 在添加了一个除法后，结果信号中的数据峰值平均偏差消失了
            H[m:n] = H[m:n] + (h - np.mean(h)) / np.std(h)

    return H


def project_pos(BGR_signal, fps, order, l):
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

    # Разделение исходного сигнала на каналы B,G,B (R, G, B)
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

    RGB = np.transpose(np.array([R_filtered, G_filtered, B_filtered]))

    H = np.zeros(num_frames)

    for n in range(num_frames - l):
        m = n - l + 1
        # Массив, содержащий часть исходных данных (от m-й до n-й строки)
        # 包含部分原始数据的数组(从m到n行)
        C = RGB[m:n, :].T
        if m >= 0:
            # Нормализация
            mean_color = np.mean(C, axis=1)
            diag_mean_color = np.diag(mean_color)
            diag_mean_color_inv = np.linalg.inv(diag_mean_color)
            Cn = np.matmul(diag_mean_color_inv, C)

            # Матрица коэффициентов
            projection_matrix = np.array([[-0.4082, -0.4082, 0.8165],
                                          [0.7071, -0.7071, 0]])

            S = np.matmul(projection_matrix, Cn)

            # Полосовая фильтрация (здесь S[0,:] и S[1,:] подобны Xs и Ys в методе CHROM)
            S0 = bandpass_filter(S[0], fps, lowcut=0.7, highcut=4.0, order=order)
            S1 = bandpass_filter(S[1], fps, lowcut=0.7, highcut=4.0, order=order)

            # Здесь S[0,:] - S1, S[1,:] - S2 по алгоритму в Wang2017_2
            std = np.array([1, np.std(S0) / np.std(S1)])
            h = np.matmul(std, S)

            # Вычисление итогового сигнала
            # Деление на np.std(h) взято с реализации в интернете
            # https://github.com/pavisj/rppg-pos/blob/master/pos_face_seg.py
            # После добавления деления на среднее отклонение всплески данных в результирующем сигнале уходят

            # 计算最终信号
            # np.std(h)除以互联网销售
            # https://github.com/pavisj/rppg-pos/blob/master/pos_face_seg.py
            # 在添加了一个除法后，结果信号中的数据峰值平均偏差消失了
            H[m:n] = H[m:n] + (h - np.mean(h)) / np.std(h)

    return H