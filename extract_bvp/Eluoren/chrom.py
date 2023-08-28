import numpy as np
from scipy.signal import butter, lfilter
from scipy.signal import get_window
from signal_filter import *



# (信号计算函数)
def chrom(signal, fps, interval_length=None):

    # (原始数据帧数)
    num_frames = len(signal)


    # (原始数据容许度检查)
    if (num_frames == 0):
        #
        # (原始数据数组是空的)
        raise NameError('EmptyData')

    # fps (fps容许值检查)
    if fps < 9:

        # (不可接受的fps值,需要fps > 9来运行带状滤波器。)
        raise NameError('WrongFPS')

     # (henning窗口宽度的容许值检查，在允许的情况下设置值)
    if (interval_length == None):
        # 在2013年的一篇文章中，Haan2013使用了20帧 / 秒的记录，窗户宽度为32帧
        # 为了保持比例，使用了32 / 20, 因数(fps=20)。
        interval_size = int(fps * (32.0 / 20))

    elif (interval_length >= 32):
        interval_size = int(interval_length // 1)

    else:
        interval_size = int(fps * (32.0 / 20))
        #
        # 亨宁窗户宽度不可接受的值，值必须至少32
        raise NameError('WrongIntervalLength')


    # 原始数据容许度检查
    if (num_frames < interval_size):
          # 不可接受的原始数据长度，原始数据长度必须至少是亨宁的窗口。
        raise NameError('NotEnoughData')


    # 将原始信号划分为R、G、B通道
    R = signal[:, 2]
    G = signal[:, 1]
    B = signal[:, 0]

    # -------------------------------------------------------------------
     # S信号间隔计算
    def S_signal_on_interval(low_limit, high_limit):

         # 分离R、G、B段间隔和标准化
        if (low_limit < 0.0):
            num_minus = abs(low_limit)
            R_interval = np.append(np.zeros(num_minus), R[0:high_limit + 1])
            R_interval_norm = R_interval / R_interval[num_minus:interval_size].mean()
            G_interval = np.append(np.zeros(num_minus), G[0:high_limit + 1])
            G_interval_norm = G_interval / G_interval[num_minus:interval_size].mean()
            B_interval = np.append(np.zeros(num_minus), B[0:high_limit + 1])
            B_interval_norm = B_interval / B_interval[num_minus:interval_size].mean()

        elif (high_limit > num_frames):
            num_plus = high_limit - num_frames
            R_interval = np.append(R[low_limit:num_frames], np.zeros(num_plus + 1))
            R_interval_norm = R_interval / R_interval[0:interval_size - num_plus - 1].mean()
            G_interval = np.append(G[low_limit:num_frames], np.zeros(num_plus + 1))
            G_interval_norm = G_interval / G_interval[0:interval_size - num_plus - 1].mean()
            B_interval = np.append(B[low_limit:num_frames], np.zeros(num_plus + 1))
            B_interval_norm = B_interval / B_interval[0:interval_size - num_plus - 1].mean()

        else:
            R_interval = R[low_limit:high_limit + 1]
            R_interval_norm = R_interval / R_interval.mean()
            G_interval = G[low_limit:high_limit + 1]
            G_interval_norm = G_interval / G_interval.mean()
            B_interval = B[low_limit:high_limit + 1]
            B_interval_norm = B_interval / B_interval.mean()


        # Xs 和 Ys成分计算
        Xs = 3.0 * R_interval_norm - 2.0 * G_interval_norm
        Ys = 1.5 * R_interval_norm + G_interval_norm - 1.5 * B_interval_norm

        # Xs = R_interval_norm - G_interval_norm
        # Ys = R_interval_norm + G_interval_norm - 2.0* B_interval_norm

           # 过滤功能调用(Xs和Ys带滤波器从0.5到4赫兹过滤)
        Xf = bandpass_filter(Xs, fps, cutoff=(0.7,4.0), order=2)
        Yf = bandpass_filter(Ys, fps, cutoff=(0.7,4.0), order=2)

         # 窗口使用前的S信号计算
        alpha = Xf.std() / Yf.std()
        S_before = Xf - alpha * Yf

        return S_before

    # -------------------------------------------------------------------

    # 搜索间隔数
    number_interval = 2.0 * num_frames / interval_size + 1
    number_interval = int(number_interval // 1)

    # 搜索区间边界并计算最终信号
    intervals = []
    S_before_on_interval = []
    for i in range(int(number_interval)):
        i_low = int((i - 1) * interval_size / 2.0 + 1)
        i_high = int((i + 1) * interval_size / 2.0)
        intervals.append([i_low, i_high])
        S_before_on_interval.append(S_signal_on_interval(i_low, i_high))


    wh = get_window('hamming', interval_size)

     # 搜索hamming窗户没有交叉点的索引
    index_without_henning = []

    for i in range(intervals[0][0], intervals[1][0], 1):
        if (i >= 0):
            index_without_henning.append(i)

    for i in range(intervals[len(intervals) - 2][1] + 1, intervals[len(intervals) - 1][1], 1):
        if (i <= num_frames):
            index_without_henning.append(i)


    S_after = np.zeros(num_frames)
    for i in range(num_frames):
        for j in intervals:
            if (i >= j[0] and i <= j[1]):
                num_interval = intervals.index(j)
                num_element_on_interval = i - intervals[num_interval][0]
                if (i not in index_without_henning):
                    S_after[i] += S_before_on_interval[num_interval][num_element_on_interval] * wh[
                        num_element_on_interval]
                else:
                    S_after[i] += S_before_on_interval[num_interval][num_element_on_interval]

    return S_after
