import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
from sklearn.preprocessing import MinMaxScaler, normalize
from scipy.signal import firwin, lfilter, medfilt, periodogram,welch
import scipy.signal as signal
from scipy.signal import find_peaks,stft
from scipy.interpolate import interp1d
import math
from pyti import catch_errors
from pyti.weighted_moving_average import weighted_moving_average as wma

import h5py
import os
import pdb
from dtw import accelerated_dtw   # 动态时间规整评价时域信号相似性


# 数据归一化 [0,1]
def min_max_norm(x):
    x = np.array(x)
    x_norm = (x-np.min(x))/(np.max(x)-np.min(x))
    return x_norm

# 数据归一化 [-1,1]
def mean_max_norm(x):
    x = np.array(x)
    x_norm = (x-np.mean(x))/(np.max(x)-np.min(x))
    return x_norm

# 数据标准化
def standar_scaler(x):
    x = np.array(x)
    m = np.mean(x)
    s = np.std(x)
    if s == 0:
        x_norm = x - m
    else:
        x_norm = (x - m) / s
    return x_norm


# 信噪比
def SignaltoNoiseRatio(Arr, axis=0, ddof=0):
    Arr = np.asanyarray(Arr)
    me = Arr.mean(axis=axis)
    sd = Arr.std(axis=axis, ddof=ddof)
    # print(me,sd)
    return np.where(sd == 0, 0, me/sd)

# 峰值信噪比
def PeakSignaltoNoiseRatio(img1, img2):
   mse = np.mean((img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# 基于fft计算 cPPG 的信噪比,计算谐波信息
def SignaltoNoiseRatio_cPPG(signal, fps, method='fft',power_max_index=None, draw_fig_flag=False):

    if method=='stft':
        # 基于stft计算信噪比
        nperseg = int(min(len(signal) * 0.5, fps * 15))
        # # print(f'nperseg {nperseg}')
        half_freqs, half_power, win = get_stft_freq_power(signal, fps, nperseg=nperseg)
    else:
        # 基于fft计算信噪比
        half_freqs, half_power, win = get_fft_freq_power(signal, fps)

    # 信号过滤 [0.7, 3.0]
    half_power[half_freqs <= 0.7] = 0
    half_power[half_freqs >= 3.0] = 0

    power_max_idx = np.argmax(half_power)

    if power_max_index is not None:
        power_max_idx = power_max_index
        # print('SignaltoNoiseRatio_rPPG power_max_idx not None', power_max_idx)

    power_main_region = half_power[int(power_max_idx - win):int(power_max_idx + win + 1)]
    freq_main_reqion = half_freqs[int(power_max_idx - win):int(power_max_idx + win + 1)]
    power_sub_region = half_power[int(power_max_idx - win)*2:int(power_max_idx + win + 1)*2]
    freq_sub_reqion = half_freqs[int(power_max_idx - win)*2:int(power_max_idx + win + 1)*2]

    signal_power = np.sum(power_main_region) + np.sum(power_sub_region)
    fft_snr = signal_power / (np.sum(half_power) - signal_power)
    # print(f'SignaltoNoiseRatio_rPPG power_max_idx, {power_max_idx}, fft_snr {fft_snr}', )

    # draw_fig_flag = True
    if draw_fig_flag:
        hr_freqs = half_freqs[power_max_idx]
        HR_est = hr_freqs * 60
        print(f'SignaltoNoiseRatio_rPPG power_max_idx, {power_max_idx}, fft_snr {fft_snr}, HR_est {HR_est}')

        plt.figure('SignaltoNoiseRatio_rPPG',figsize=(9,6))
        plt.subplot((211))
        plt.plot(np.linspace(0,len(signal)/fps,len(signal)),signal,label='HR '+ str(np.round(HR_est,2)))
        plt.legend(loc='upper right')

        plt.subplot((212))
        plt.plot(half_freqs[half_freqs < 5], half_power[half_freqs < 5])
        plt.scatter(hr_freqs, np.max(half_power), label='hr_freqs ' + str(np.round(hr_freqs, 2)))
        plt.vlines(hr_freqs, 0, int(np.max(half_power) * 1.2), colors='y', linestyles='dashed')
        plt.plot(freq_main_reqion, power_main_region,'r',label='SNR ' + str(np.round(fft_snr,2)))
        plt.plot(freq_sub_reqion, power_sub_region, 'r', label='SNR ' + str(np.round(fft_snr, 2)))
        plt.legend(loc='upper right')
        plt.show()

    return fft_snr


# 基于stft计算 rPPG 的信噪比，因谐波多被掩盖，不计算谐波信息
def SignaltoNoiseRatio_rPPG(signal, fps, power_max_index=None, method='fft', draw_fig_flag=False):

    if method=='stft':
        # 基于stft计算信噪比
        nperseg = int(min(len(signal) * 0.5, fps * 15))
        # # print(f'nperseg {nperseg}')
        half_freqs, half_power, win = get_stft_freq_power(signal, fps, nperseg=nperseg)

    elif method=='fft':
        # 基于fft计算信噪比
        half_freqs, half_power, win = get_fft_freq_power(signal, fps)

    elif method == 'welch':
        # # welch傅里叶变换
        nperseg = int(min(len(signal) * 0.5, fps * 15))
        # nperseg = len(signal)
        half_freqs, half_power = welch(signal, fps, 'flattop', nperseg=nperseg, average='median')
        # 预测心率误差 5 以内的频域窗
        win = int(len(half_freqs) / (12 * np.max(half_freqs)) + 1)
    else:
        print(f'Warning: FFT method not defined')
        half_freqs, half_power = None, None
        pdb.set_trace()

        # 信号过滤 [0.7, 3.0]
    half_power[half_freqs <= 0.7] = 0
    half_power[half_freqs >= 3.0] = 0

    power_max_idx = np.argmax(half_power)

    if power_max_index is not None:
        power_max_idx = power_max_index
        # print('SignaltoNoiseRatio_rPPG power_max_idx not None', power_max_idx)

    power_main_region = half_power[int(power_max_idx - win):int(power_max_idx + win + 1)]
    freq_main_reqion = half_freqs[int(power_max_idx - win):int(power_max_idx + win + 1)]

    signal_power = np.sum(power_main_region)
    fft_snr = signal_power / (np.sum(half_power) - signal_power)
    # print(f'SignaltoNoiseRatio_rPPG power_max_idx, {power_max_idx}, fft_snr {fft_snr}', )

    # draw_fig_flag = True
    if draw_fig_flag:
        hr_freqs = half_freqs[power_max_idx]
        HR_est = hr_freqs * 60
        print(f'SignaltoNoiseRatio_rPPG method {method}, power_max_idx, {power_max_idx},'
              f' fft_snr {fft_snr}, HR_est {HR_est}')

        plt.figure('SignaltoNoiseRatio_rPPG',figsize=(9,6))
        plt.subplot((211))
        plt.plot(np.linspace(0,len(signal)/fps,len(signal)),signal,label='HR '+ str(np.round(HR_est,2)))
        plt.legend(loc='upper right')

        plt.subplot((212))
        plt.plot(half_freqs[half_freqs < 5], half_power[half_freqs < 5])
        plt.scatter(hr_freqs, np.max(half_power), label='hr_freqs ' + str(np.round(hr_freqs, 2)))
        plt.vlines(hr_freqs, 0, int(np.max(half_power) * 1.2), colors='y', linestyles='dashed')
        plt.plot(freq_main_reqion, power_main_region,'r',label='SNR ' + str(np.round(fft_snr,2)))
        plt.legend(loc='upper right')
        plt.show()

    return fft_snr






# 数据降维
def npydata_reduction_fn(file_path):
    data = np.load(file_path)

    new_data = np.zeros((data.shape[0], data.shape[2]))
    for i in range(data.shape[0]):
        new_data[i, :] = np.squeeze(data[i, :, :].mean(axis=0))

    # print(f'data {data.shape} \n{data[0]}')
    # pdb.set_trace()
    return new_data


# 基于三次立方插值法进行数据重采样
def Cubic_Interpolation(signals, fps, time_points=30, time_duration='None'):
    # cPPG and rPPG signals
    if time_duration != 'None':
        time_duration = time_duration
    else:
        signals = np.squeeze(signals)
        time_duration = len(signals) / fps

    signal_num = int(time_duration * time_points)
    t0 = np.linspace(0, time_duration, len(signals))
    t1 = np.linspace(0, time_duration, signal_num)
    # draw_1_signal(signals)
    # print(len(t0),len(signals))
    interpolation = interp1d(t0, signals, kind='cubic',bounds_error=True)
    resample_signals = interpolation(t1)

    # draw_2_figure(signals, resample_signals, legend1='signals', legend2='resample_signals')
    # pdb.set_trace()
    return resample_signals


# exist_file_names = check_exist_files(path)
def check_exist_files(path):

    if os.path.exists(path):

        file_list = os.listdir(path)

        exist_file_names = []
        for file in file_list:
            file_name = file.split('.')[0]
            # file_name = '_'.join(file.split('_')[:3])
            exist_file_names.append(file_name)
    else:
        exist_file_names = []

    return exist_file_names



# 构建数据切分的窗口
def CreateWindows(data, fs, win_time, step_time=1, overlap=None):
    data = np.array(data)
    win_len = int(win_time * fs)
    N_samp = data.shape[0] - win_len + 1
    if overlap is not None:
        step_len = int(win_len * (1 - overlap))
    else:
        step_len = int(step_time * fs)
    idx_start = np.round(np.arange(0, N_samp, step_len)).astype(int)
    idx_stop = np.round(idx_start + win_len)
    return idx_start, idx_stop

# 构建数据切分的窗口
def CreateWindows_bk(N_samp, win_time, step_time, fs):
    win_len = int(win_time * fs)
    step_len = int(step_time * fs)
    N_samp_ = N_samp - win_len + 1
    idx_start = np.round(np.arange(0, N_samp_, step_len)).astype(int)
    idx_stop = np.round(idx_start + win_len)
    return idx_start, idx_stop

# check and fill nan or inf in ndarray.
def fill_ndarray(t1):
    t1 = np.array(t1)
    ret = False
    try:
        for i in range(t1.shape[1]):
            temp_col = t1[:,i]
            nan_num = np.count_nonzero(temp_col != temp_col)
            if nan_num != 0:
                ret = True
                temp_not_nan_col = temp_col[temp_col == temp_col]
                temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()
    except:
        temp_col = t1
        nan_num = np.count_nonzero(temp_col != temp_col)
        print(f'nan_num {nan_num}')
        if nan_num != 0:
            ret = True
            temp_not_nan_col = temp_col[temp_col == temp_col]
            temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()
    return ret, t1


# 傅里叶分析，预测心率值
def fourier_analysis(signal, fps, draw_figures_flag=False):
    def find_idx_bigger(arr, thr):
        return next(f[0] for f in enumerate(arr) if f[1] > thr)

    MinFreq = 42  # bpm
    MaxFreq = 180  # bpm
    freqs, psd = periodogram(signal, fs=fps, window=None, detrend='constant', return_onesided=True,
                             scaling='density')
    min_idx = find_idx_bigger(freqs, MinFreq / 60.0) - 1
    max_idx = find_idx_bigger(freqs, MaxFreq / 60.0) + 1
    freq_max = freqs[min_idx + np.argmax(psd[min_idx: max_idx])]

    hr_estimated = freq_max * 60

    def draw_figure(freqs,psd,freq_max):
        freqs = freqs[freqs<4]
        psd = psd[:len(freqs)]
        plt.figure(figsize=(9,6))
        plt.plot(freqs,psd,color='r')
        plt.axvline(x=freq_max, c="b", ls="--", lw=2)
        plt.title('fourier_analysis')
        plt.show()

    if draw_figures_flag:
        draw_figure(freqs,psd,freq_max)

    return hr_estimated

# # 快速傅里叶变换提取 half_freq, half_power
def get_fft_freq_power(signal, fps, return_onesided=True):
    # 数据去趋势,及归一化
    signal = scipy.signal.detrend(signal)
    signal = standar_scaler(signal)

    signal_fft = np.fft.fft(signal)
    power = np.abs(signal_fft) ** 2
    freqs = np.fft.fftfreq(signal_fft.size, 1 / fps)  # 得到分解波的频率序列

    # 预测心率误差 5 以内的频域窗
    win = int(len(freqs[0 <= freqs]) / (12 * np.max(freqs)) + 1)

    if return_onesided:
        power = power[0 <= freqs]
        freqs = freqs[0 <= freqs]

    return freqs, power, win


# # 快速傅里叶变换提取 half_freq, half_power
def get_stft_freq_power(signal, fps, method='median', return_onesided=True, nperseg=256):
    # 数据去趋势,及归一化
    signal = scipy.signal.detrend(signal)
    signal = standar_scaler(signal)

    # # 短时傅里叶变换
    freqs, times, powers = stft(signal, fs=fps, nperseg=nperseg, return_onesided=return_onesided)
    powers = np.abs(powers) ** 2

    # 预测心率误差 5 以内的频域窗
    win = int(len(freqs[0 <= freqs]) / (12 * np.max(freqs)) + 1)

    if method == 'median':
        median_powers = np.median(powers, axis=1)
        return freqs, median_powers, win

    elif method == 'mean':
        mean_powers = np.mean(powers, axis=1)
        return freqs, mean_powers, win

    # plt.figure('get_stft_freq_power',figsize=(9,6))
    # plt.plot(half_freqs,half_powers)
    # plt.show()
    # pdb.set_trace()



def get_stft_maxidx(signal, fps, win=None):

    # 数据标准化,及归一化
    signal = mean_max_norm(signal)
    signal = standar_scaler(signal)
    # 傅里叶变换
    signal_fft = np.fft.fft(signal)
    power = np.abs(signal_fft) ** 2
    freqs = np.fft.fftfreq(signal_fft.size, 1 / fps)  # 得到分解波的频率序列
    half_power = power[0 <= freqs]
    half_freqs = freqs[0 <= freqs]

    half_power[half_freqs <= 0.7] = 0
    half_power[half_freqs >= 2.5] = 0

    if win is None:
        win = int(len(half_freqs) / (12 * np.max(half_freqs)) + 1)
        # print(f'get_stft_maxidx win {win}')

    # 默认最大频谱索引值
    power_max_default = np.argmax(half_power)
    HR_default = half_freqs[power_max_default] * 60

    power_index_top = list(np.argsort(-half_power)[:5])
    power_index_top_mean = np.mean(half_power[power_index_top])
    # print('power_index_top_mean',power_index_top_mean)
    # print('power_index_top',power_index_top)

    # # 短时傅里叶变换，计算心率值和频谱值

    HR_stft_error_flag = 10
    power_sum_flag = 0
    power_stft_med_idx = None
    power_sum_idx = None
    HR_sum = None
    HR_med = None
    ret = False
    for i in power_index_top:
        # # 默认主频信号高于其他信号的2倍，则确定默认频谱为心率频谱
        if half_power[i] >= power_index_top_mean * 2:
            power_max_default = i
            power_sum_idx = i
            # HR_default = half_freqs[power_max_default] * 60
            ret = True
            break

        up = int(i + win) + 1
        low = max(int(i - win), 0)
        power_sum = np.sum(half_power[low:up])
        # 基于主频域筛选
        if power_sum >= power_sum_flag:
            power_sum_flag = power_sum
            # power_sum_idx = low + np.argmax(half_power[low:up])
            power_sum_idx = i
            HR_sum = half_freqs[power_sum_idx] * 60
            # print(f'{i}, power_sum_idx {power_sum_idx}, HR_freq {half_freqs[power_sum_idx]}, HR {HR_sum}')

        temp_HR = half_freqs[i]*60
        # 基于短时傅里叶变化的中值频域筛选
        HR_med_error = abs(HR_stft_med - temp_HR)

        if HR_stft_error_flag >= HR_med_error:
            HR_stft_error_flag = HR_med_error

            power_stft_med_idx = i
            HR_med = half_freqs[power_stft_med_idx] * 60

    power_max_idx = None
    if ret:
        power_max_idx = power_max_default
    elif power_stft_med_idx is not None:
        power_idx_list = [power_max_default, power_stft_med_idx, power_sum_idx]

        HR_error_flag = 20
        temp_ret = True
        for j in power_idx_list:
            temp_HR_ = half_freqs[j] * 60
            HR_error = abs(temp_HR_ - 80)
            if HR_error_flag >= HR_error:
                HR_error_flag = HR_error
                power_max_idx = j
                temp_ret = False

        if temp_ret:
            power_max_idx = np.sort(power_idx_list)[1]
    else:
        power_max_idx = power_sum_idx

    sub_win = int(win/2)
    half_power = fill_zero_nainf(half_power)
    if sub_win != 0:
        power_max_idx = (power_max_idx - sub_win) + np.argmax(half_power[power_max_idx - sub_win:power_max_idx + sub_win + 1])

    if power_max_idx is None:
        power_max_idx = power_max_default


    return power_max_idx

# 筛选合适的频谱值，异常噪音可能掩盖心电信号，通过临近频谱进行二次筛选
def get_power_maxidx(signal, fps, method='fft', draw_subfig_flag=False):

    def get_main_region(power_max_index):
        power_main_region = half_power[int(power_max_index - win):int(power_max_index + win + 1)]
        freq_main_reqion = half_freqs[int(power_max_index - win):int(power_max_index + win + 1)]
        signal_power = np.sum(power_main_region)
        fft_snr = signal_power / (np.sum(half_power) - signal_power)
        # power_sub_region = half_power[int(power_max_index - win) * 2:int(power_max_index + win + 1) * 2]
        # freq_sub_reqion = half_freqs[int(power_max_index - win) * 2:int(power_max_index + win + 1) * 2]
        return power_main_region, freq_main_reqion, fft_snr

    if method=='stft':
        # 基于stft计算信噪比
        nperseg = int(min(len(signal) * 0.5, fps * 15))
        # # print(f'nperseg {nperseg}')
        half_freqs, half_power, win = get_stft_freq_power(signal, fps, nperseg=nperseg)
    else:
        # 基于fft计算信噪比
        half_freqs, half_power, win = get_fft_freq_power(signal, fps)

    # 去除异常频谱
    half_power[half_freqs > 3.0] = 0
    half_power[half_freqs < 0.7] = 0

    # plt.figure('get_power_maxidx',figsize=(9,6))
    # plt.plot(half_freqs,half_power)
    # plt.show()
    # pdb.set_trace()

    # 获取获取最大频谱索引值
    power_max_default = np.argmax(half_power)
    # HR_default = half_freqs[power_max_default] * 60
    power_index_top = list(np.argsort(-half_power)[:5])
    power_index_top_mean = np.mean(half_power[power_index_top])
    # print('power_index_top_mean',power_index_top_mean)

    power_sum_flag = 0
    power_idx = None
    for i in power_index_top:
        up = int(i + win) + 1
        low = max(int(i - win),0)

        if half_power[power_max_default] >= power_index_top_mean * 2:
            power_idx = None
            break

        power_sum = np.sum(half_power[low:up])
        if power_sum >= power_sum_flag:
            power_sum_flag = power_sum
            # power_idx = low + np.argmax(half_power[low:up])
            power_idx = i
            # HR_sum = half_freqs[power_idx] * 60
            # print(f'{i}, power_sum_idx {power_idx}, power_sum_flag {power_sum_flag},'
            #       f'HR_freq {half_freqs[power_idx]}, HR {HR_sum}')

        if draw_subfig_flag:
            power_main_region, freq_main_reqion, fft_snr = get_main_region(i)
            print(f'{i}, power_idx {power_idx}, fft_snr {fft_snr}'
                  f'HR_freq {half_freqs[i]}, HR {half_freqs[i]*60}')

            plt.figure(figsize=(9,6))
            plt.subplot((211))
            plt.plot(half_freqs[half_freqs < 5.0], half_power[half_freqs < 5.0])
            plt.plot(freq_main_reqion, power_main_region,c='r',
                     label='fft_snr ' + str(np.round(fft_snr, 2))
                     )
            plt.scatter(half_freqs[i], half_power[i], alpha=0.5, label='HR ' + str(np.round(half_freqs[i]*60, 1)))
            plt.legend(loc='upper right')

            plt.subplot((212))
            plt.plot(freq_main_reqion, power_main_region, c='r',
                     label='freq win ' + str(np.round(half_freqs[low], 2)) + '_' + str(np.round(half_freqs[up], 2))
                     )
            plt.scatter(half_freqs[i], half_power[i], alpha=0.5, label='HR freqs ' + str(np.round(half_freqs[i], 1)))
            plt.vlines(half_freqs[i], -1, int(np.max(half_power) * 1.2), colors='r', linestyles='dashed')
            plt.legend(loc='upper right')
            plt.show()
            plt.close()

    if power_idx is not None:
        # HR_sum = half_freqs[power_idx] * 60
        # if abs(HR_default - 80) > 20 and abs(HR_sum - 80) <= 20:
        power_max_idx = (power_idx - win) + np.argmax(half_power[(power_idx - win):(power_idx + win +1)])
    else:
        power_max_idx = power_max_default

    # print(f'power_max_idx {power_max_idx},  freqs {half_freqs[power_idx]},  HR {half_freqs[power_idx]*60}'
    #       f'power_max_default {power_max_default},  freqs_default {half_freqs[power_max_default]}, '
    #       f' HR_default {half_freqs[power_max_default]*60}'
    #       )

    return power_max_idx


# 逆变换之前处理异常值 nan 和 inf,替换为 0
def fill_zero_nainf(signal_fft_new):
    if np.isinf(signal_fft_new).any() or np.isnan(signal_fft_new).any():
        inf_ = np.isinf(signal_fft_new)
        signal_fft_new[inf_] = 0
        nan_ = np.isnan(signal_fft_new)
        signal_fft_new[nan_] = 0
        print(f'nan or inf in signal_fft_new')
        # print(f'inf_ {inf_}')
        # print(f'nan_ {nan_}')
        # pdb.set_trace()
    return signal_fft_new


# 增强心电信号的频谱，消减非心电信号的频谱，与心电信号的频谱相近越大，消减越明显
def enhance_signal_fft(signal, fps, power_max_index=None, item=1):
    # # 平滑处理可以过滤重搏波
    # signal = hull_moving_average(signal, window_size=5)
    # 数据标准化,及归一化
    # signal = mean_max_norm(signal)
    signal = standar_scaler(signal)
    # 傅里叶变换
    signal_fft = np.fft.fft(signal)
    signal_fft = fill_zero_nainf(signal_fft)
    signal_fft_ori = signal_fft
    power = np.abs(signal_fft) ** 2
    freqs = np.fft.fftfreq(signal_fft.size, 1 / fps)  # 得到分解波的频率序列
    half_power = power[0 <= freqs]
    half_freqs = freqs[0 <= freqs]

    snr = SignaltoNoiseRatio_rPPG(signal, fps)

    if power_max_index is not None:
        power_max_idx = power_max_index
    else:
        power_max_idx = get_stft_maxidx(signal, fps)
    # print(f'enhance_signal_fft power_max_idx {power_max_idx} ')
    power_index_list = np.argsort(-half_power[half_freqs < 5.0]).tolist()
    # print(f'enhance_signal_fft power_index_list {len(power_index_list)},{power_index_list} ')
    if item == 1:
        w = max(abs(1/snr),3)
        # print(f'enhance_signal_fft power_max_idx {power_max_idx} w {w} ',
              # signal_fft[power_max_idx]
              # )
        signal_fft[power_max_idx] = signal_fft[power_max_idx] * w

        signal_fft[len(signal_fft) - power_max_idx] = signal_fft[len(signal_fft) - power_max_idx] * w

        for i in range(int(len(power_index_list) * 0.1)):
            if (power_max_idx - power_index_list[i]) != 0 and abs(power_max_idx*2 - power_index_list[i])>2:
                signal_fft[power_index_list[i]] = signal_fft[power_index_list[i]] / (abs(power_max_idx - power_index_list[i]))
                signal_fft[len(signal_fft) - power_index_list[i]] = signal_fft[power_index_list[i]] / (abs(
                    power_max_idx - power_index_list[i]))

    elif item == 2:
        signal_fft[power_max_idx] = signal_fft[power_max_idx] **2
        signal_fft[len(signal_fft) - power_max_idx] = signal_fft[len(signal_fft) - power_max_idx] **2

        for i in range(int(len(power_index_list) * 0.1)):
            if (power_max_idx - power_index_list[i]) != 0 and abs(power_max_idx*2 - power_index_list[i])>2:
                signal_fft[power_index_list[i]] = signal_fft[power_index_list[i]] / (abs(power_max_idx - power_index_list[i])**2)
                signal_fft[len(signal_fft) - power_index_list[i]] = signal_fft[power_index_list[i]] / (abs(
                    power_max_idx - power_index_list[i])**2)

    signal_fft = fill_zero_nainf(signal_fft)
    # print(f'enhance_signal_fft signal_fft_ori {signal_fft_ori} ')
    # print(f'enhance_signal_fft signal_fft {signal_fft} ')

    # draw_2_signals(np.real(signal_fft_ori),np.real(signal_fft),title='enhance_signal_fft')
    return signal_fft


# # 增强心电信号的频谱，消减非心电信号的频谱，与心电信号的频谱相近越大，消减越明显
# def enhance_signal_fft(freqs, power, power_max_idx, snr, item=1):
#     # # 平滑处理可以过滤重搏波
#     # signal = hull_moving_average(signal, window_size=5)
#     # 数据标准化,及归一化
#     # signal = mean_max_norm(signal)
#     # signal = standar_scaler(signal)
#     # 傅里叶变换
#     # power = np.fft.fft(signal)
#     # power = fill_zero_nainf(power)
#     power_ori = power
#     half_power = power[0 <= freqs]
#     half_freqs = freqs[0 <= freqs]
#
#     # print(f'enhance_signal_fft power_max_idx {power_max_idx} ')
#     # power_index_list = np.argsort(-half_power[half_freqs < 5.0]).tolist()
#     # draw_2_signals(freqs, power)
#     # plt.plot(freqs, power)
#     # plt.show()
#
#     power_index_list = np.argsort(power[abs(freqs) < 5.0]).tolist()
#     print(f'enhance_signal_fft power_index_list {len(power_index_list)}, {power_index_list} ')
#
#     if item == 1:
#         w = max(abs(2/snr),10)
#         # w =1
#         print(f'enhance_signal_fft power_max_idx {power_max_idx} w {w} ',
#               power[power_max_idx]
#               )
#         power[power_max_idx] = power[power_max_idx] * w
#
#         power[len(power) - power_max_idx] = power[len(power) - power_max_idx] * w
#
#         # for i in range(int(len(power_index_list) * 0.1)):
#         #     if (power_max_idx - power_index_list[i]) != 0 and abs(power_max_idx*2 - power_index_list[i])>2:
#         #         power[power_index_list[i]] = power[power_index_list[i]] / (abs(power_max_idx - power_index_list[i]))
#         #         power[len(power) - power_index_list[i]] = power[power_index_list[i]] / (abs(
#         #             power_max_idx - power_index_list[i]))
#
#     elif item == 2:
#         power[power_max_idx] = power[power_max_idx] **2
#         power[len(power) - power_max_idx] = power[len(power) - power_max_idx] **2
#
#         for i in range(int(len(power_index_list) * 0.1)):
#             if (power_max_idx - power_index_list[i]) != 0 and abs(power_max_idx*2 - power_index_list[i])>2:
#                 power[power_index_list[i]] = power[power_index_list[i]] / (abs(power_max_idx - power_index_list[i])**2)
#                 power[len(power) - power_index_list[i]] = power[power_index_list[i]] / (abs(
#                     power_max_idx - power_index_list[i])**2)
#
#     power = fill_zero_nainf(power)
#     # print(f'enhance_signal_fft power_ori {power_ori} ')
#     # print(f'enhance_signal_fft power {power} ')
#
#     # draw_2_signals(np.real(power_ori),np.real(power),title='enhance_signal_fft')
#     return power




# 峰值检测，预测心率值
def find_peaks_analysis(signals, fps, draw_figures_flag=False):
    signals = signal.detrend(signals)
    sample_count = len(signals)
    duration = int(sample_count / fps)
    time_line = np.linspace(0, duration, sample_count)  # 采样点的时间

    test_peaks = min(300,int(sample_count * 0.1))
    # 主峰检测
    down_peak = np.sort(signals)[int(test_peaks * 0.3):int(test_peaks * 0.7)]  # sort默认从小到大排序，前300个峰值为30-90秒内的下部峰值点
    up_peak = np.sort(signals)[-int(test_peaks * 0.7):-int(test_peaks * 0.3)]  # sort默认从小到大排序，后300个峰值为30-90秒内的上部峰值点
    median_peak = np.median(signals)
    down_height = np.abs(down_peak - median_peak)  # 下部峰值点与均值的差距
    up_height = np.abs(up_peak - median_peak)  # 上部峰值点与均值的差距
    main_peak = np.sum(up_height) - np.sum(down_height)  # 上部峰值点与均值的差距与下部峰值点与均值的差距的差值

    # print('mean_height', mean_height)
    # print('up_peak', len(up_peak), up_peak)
    # print('down_peak', len(down_peak), down_peak)
    # print(down_height, up_height)
    # print(main_peak)
    # pdb.set_trace()

    def main_peak_test(mean_peak, main_peak, up_height, down_height):
        if main_peak >= 0:
            # peaks是峰值的索引值，properties是设置的参数
            mean_height = np.median(up_height) * 0.5 + mean_peak
            # print('np.mean(up_height): \n', np.mean(up_height))
            # print('mean_height1: \n', mean_height)
            # height=[min_height, 3*mean_height]
            peaks, properties = signal.find_peaks(signals, height=mean_height,distance=int(fps*0.5))
            # print('peaks1: \n', peaks)
            # print('properties: \n', properties)
            return peaks, properties

        elif main_peak < 0:
            mean_height = np.median(down_height) * 0.5 - mean_peak
            # print('mean_height2: \n', mean_height)
            peaks, properties = signal.find_peaks(-signals, height=mean_height)
            # print('peaks2: \n', peaks)
            return peaks, properties
            # print('properties: \n', properties)

    def draw_figure(signals,peaks,properties):
        plt.figure(figsize=(9,6))
        plt.plot(signals,color='r')
        plt.scatter(peaks,properties['peak_heights'],color='b')
        plt.title('find_peaks_analysis')
        plt.show()

    peaks, properties = main_peak_test(median_peak, main_peak, up_height, down_height)

    if draw_figures_flag:
        draw_figure(signals, peaks, properties)
    # print('peaks: \n', peaks)
    # print('properties: \n', properties)

    peaks_height = []  # 峰值点的振幅
    peaks_time = []  # 峰值点的时间

    for tp, h in enumerate(signals):
        time_peak = time_line[tp]

        if tp in peaks:
            if len(peaks_time) == 0:
                peaks_height.append(h)
                peaks_time.append(time_peak)

            else:
                time_duration = float(time_peak) - float(peaks_time[-1])
                if time_duration >= 0.5:
                    peaks_height.append(h)
                    peaks_time.append(time_peak)

                if time_duration < 0.5 and abs(float(peaks_height[-1])) < abs(float(h)):
                    peaks_height.pop()
                    peaks_time.pop()
                    peaks_height.append(h)
                    peaks_time.append(time_peak)
    # 相邻峰值之间的时间差
    delta_time = np.array(peaks_time[1:]) - np.array(peaks_time[:-1])
    # print('delta_time: \n', list(delta_time))
    hr = 60 / delta_time  # 瞬时心率
    return hr


def max_min_scaler(data):
    try:
        data_scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = data_scaler.fit_transform(data)
    except:
        data = np.expand_dims(data,axis=1)
        data_scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = data_scaler.fit_transform(data)
    return np.squeeze(data_scaled)


def fir_filter(signals, filter_coef=40, cutoffFreq=32, fs=60):
    # cPPG filter
    a = 1  # 分母
    # fs is sampling frequency of the signals
    # fs is 255 in article
    # cutoffFreq is 32 HZ in article

    b = firwin(filter_coef, cutoff=cutoffFreq, pass_zero='lowpass', fs=fs)  # FIR with low-pass
    filter_sig = lfilter(b, a, signals)

    return filter_sig


def moving_average(signals, window_size=30):
    # windows_size is 300 in article
    # vec = np.cumsum(np.insert(signals,0,0))
    # ma_vec = (vec[windows_size:]-vec[:-windows_size])/windows_size
    signals = list(signals)
    am_signals = np.convolve(signals, np.ones(window_size, dtype=int) / window_size, 'valid')
    return am_signals


    # r = np.arange(1,windows_size-1,2)
    # signals = list(signals)
    # start = np.cumsum(signals[:windows_size-1])  # [::2]/r
    # print(start[::2].shape)
    #
    # start = start[::2]/r
    # stop = (np.cumsum(signals[:-windows_size:-1])[::2]/r)[::-1]
    # am_signals = np.concatenate((start,op,stop))
    #
    # return am_signals

# yy = smooth(y) smooths the data in the column vector y ..
# The first few elements of yy are given by
# yy(1) = y(1)
# yy(2) = (y(1) + y(2) + y(3))/3
# yy(3) = (y(1) + y(2) + y(3) + y(4) + y(5))/5
# yy(4) = (y(2) + y(3) + y(4) + y(5) + y(6))/5
# ...

def smooth(data, window_size):
    # data:原始数据，NumPy 1-D array containing the data to be smoothed
    # 必须是1-D的，如果不是，请使用 np.ravel()或者np.squeeze()转化
    # window_size: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    data = np.squeeze(data)
    out0 = np.convolve(data, np.ones(window_size, dtype=int), 'valid') / window_size
    r = np.arange(1, window_size - 1, 2)
    start = np.cumsum(data[:window_size - 1])[::2] / r
    stop = (np.cumsum(data[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((start, out0, stop))

# another one，边缘处理的不好

def moving_avg_2(data, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(data, window, 'same')


# another one，速度更快
# 输出结果 不与原始数据等长，假设原数据为m，平滑步长为t，则输出数据为m-t+1
def moving_avg_3(data, window_size):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    ma_vec = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
    return ma_vec


def hull_moving_average(data, window_size):
    """
    Hull Moving Average.

    Formula:
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    """
    catch_errors.check_for_period_error(data, window_size)
    hma = wma(2 * wma(data, int(window_size/2)) - wma(data, window_size), int(np.sqrt(window_size)))
    # print(list(hma))
    nan_idx = np.isnan(hma).sum()
    hma[:nan_idx]=data[:nan_idx]
    # print(nan_idx)
    # print(list(hma))
    # nan_idx = np.isnan(hma).sum()
    # print(nan_idx)
    return hma

# "生成巴特沃斯带通滤波器"
# refer to: https://blog.csdn.net/SeaBiscuitUncle/article/details/103943900
def butterBandPassFilter(signals, lowcut, highcut, fps, order):
    semiSampleRate = fps*0.5
    low = lowcut / semiSampleRate
    high = highcut / semiSampleRate
    b,a = signal.butter(order,[low,high],btype='bandpass')
    filter_sig = lfilter(b, a, signals)
    return filter_sig

# "生成巴特沃斯带阻滤波器"
def butterBandStopFilter(signals,lowcut, highcut, fps, order):
    semiSampleRate = fps*0.5
    low = lowcut / semiSampleRate
    high = highcut / semiSampleRate
    b,a = signal.butter(order,[low,high],btype='bandstop')
    filter_sig = lfilter(b, a, signals)
    return filter_sig

# "生成巴特沃斯低通滤波器"
def butterLowpassFilter(signals,lowcut, fps, order):
    semiSampleRate = fps*0.5
    low = lowcut / semiSampleRate
    b,a = signal.butter(order,[low],btype='lowpass')
    filter_sig = lfilter(b, a, signals)
    return filter_sig

def find_signal_peaks(signal, fps, draw_figure=False):
    # find_peaks
    dur_time = len(signal) / fps
    time_line = np.linspace(0, dur_time, len(signal))
    indices_up = find_peaks(signal, height=None, threshold=None, distance=5, prominence=None, width=None, wlen=None,
                            rel_height=None, plateau_size=None)
    indices_down = find_peaks(-signal, height=None, threshold=None, distance=5, prominence=None, width=None, wlen=None,
                              rel_height=None, plateau_size=None)
    indices = indices_up + indices_down

    # argrelextrema
    # indices_up = argrelextrema(signal, np.greater)
    # indices_down = argrelextrema(-signal, np.greater)
    # indices = indices_up + indices_down
    # print(indices)

    time_peak = time_line[indices[0]]

    if draw_figure:
        plt.figure(figsize=(10, 6))
        plt.plot(time_line, signal)
        plt.plot(time_peak, signal[indices[0]], 'o')
        plt.show()

        # plt.plot(time_line[indices_up[0]], signal[indices_up], 'o')
        # plt.plot(time_line[indices_down[0]], signal[indices_down], '+')
        # plt.show()

    return time_peak

# 信号的特征提取
# refer to： https://blog.csdn.net/qq_34705900/article/details/88389319
def Extract_Signal_Features(signals,fps,time_points):
    signals = np.squeeze(signals)
    # 均值
    df_mean = signals.mean()
    # 方差
    df_var = signals.var()
    # 标准差
    df_std = signals.std()
    # 均方根
    df_rms = math.sqrt(pow(df_mean, 2) + pow(df_std, 2))
    # 偏度
    df_skew = pd.Series(signals).skew()
    # 峭度
    df_kurt = pd.Series(signals).kurt()

    sum = 0
    for i in range(signals.shape[0]):
        sum += math.sqrt(abs(signals[i]))

    # 波形因子
    df_boxing = df_rms / (abs(signals).mean())

    # 峰值因子
    df_fengzhi = (max(signals)) / df_rms

    # 脉冲因子
    df_maichong = (max(signals)) / (abs(signals).mean())

    # 裕度因子
    df_yudu = (max(signals)) / pow((sum / signals.shape[0]), 2)

    # 一阶导数
    dur_time = signals.shape[0]/fps
    timeline = np.linspace(0, dur_time, signals.shape[0])

    delta_h1 = np.squeeze(signals)[1:] - np.squeeze(signals)[:-1]
    delta_t1 = timeline[1:] - timeline[:-1]
    signal_diff1 = np.squeeze(delta_h1) / delta_t1

    signal_diff1_30 = Cubic_Interpolation(signal_diff1, fps, time_points=time_points)
    # print(f'signal_diff1:{signal_diff1.shape} ')

    # 二阶导数
    delta_h2 = signal_diff1[1:] - signal_diff1[:-1]
    delta_t2 = timeline[1:-1] - timeline[:-2]
    signal__diff2 = delta_h2 / delta_t2
    signal_diff2_30 = Cubic_Interpolation(signal__diff2, fps, time_points=time_points)

    #
    signal_features_list = [df_mean, df_var, df_rms, df_skew, df_kurt, df_boxing, df_fengzhi, df_maichong, df_yudu,
                        list(signal_diff1_30), list(signal_diff2_30)]

    # df = pd.DataFrame(signal_features_list).T
    # print('signal_features',df)
    # pdb.set_trace()
    return signal_features_list

# 查看已存在文件名称
def Check_exist_file(path):
    file_name_list = []
    if os.path.exists(path):
        file_list = os.listdir(path)
        for file in file_list:
            file_name_list.append(file.split('.')[0])
    return file_name_list

# 中值滤波器，去除基线漂移
def signal_detrend_mdf(signal, fps, draw_fig_flag=False):
    # draw_fig_flag = True

    fliter = int(fps)
    Give_up_size = int(fliter / 2)

    if fliter % 2 != 1:
        kenerl_size = fliter + 1
    else:
        kenerl_size = fliter

    Signal_baseline = medfilt(signal, kenerl_size)
    Filtered_Signal = signal - Signal_baseline
    Filtered_Signal[:Give_up_size] = signal[:Give_up_size]
    Filtered_Signal[-Give_up_size:] = signal[-Give_up_size:]
    # Final_Filtered_Signal = Filtered_Signal[Give_up_size: -Give_up_size]

    if draw_fig_flag:
        plt.figure('signal detrend mdf',figsize=(9, 6))
        plt.subplot(4, 1, 1)
        plt.ylabel("Original signal")
        plt.plot(signal[Give_up_size:-Give_up_size])  # 输出原图像
        plt.subplot(4, 1, 2)
        plt.ylabel("Original baseline")
        plt.plot(Signal_baseline[Give_up_size:-Give_up_size])  # 输出基线轮廓
        plt.subplot(4, 1, 3)
        plt.ylabel("Filtered signal")
        plt.plot(Filtered_Signal)  # 基线滤波结果
        plt.subplot(4, 1, 4)
        plt.ylabel("Filtered signal baseline")
        Filtered_ECG_signal_baseline = medfilt(Filtered_Signal, kenerl_size)
        plt.plot(Filtered_ECG_signal_baseline[Give_up_size:-Give_up_size])  # 基线滤波结果
        plt.show()

    return Filtered_Signal

# rPPG processing
def Single_rPPG_Processing(single_rPPG, fps):
    single_rppg_scl = max_min_scaler(single_rPPG)
    filter_rppg_mdf = signal_detrend_mdf(single_rppg_scl, fps, draw_fig_flag=False)
    filter_rPPG_btf = butterBandPassFilter(filter_rppg_mdf, lowcut=0.7, highcut=3.0, fps=fps, order=6)
    filter_rPPG_hma = hull_moving_average(filter_rPPG_btf, window_size=int(0.3*fps))

    # draw_figures(filter_rppg_mdf, filter_rPPG_btf, filter_rPPG_hma, legend1='filter_rppg_mdf', legend2="filter_rPPG_btf",
    #                  legend3='filter_rPPG_hma')

    return filter_rPPG_hma

# cPPG processing
def Single_cPPG_Processing(single_cPPG, fps):
    single_cppg_scl = max_min_scaler(single_cPPG)
    filter_cppg_mdf = signal_detrend_mdf(single_cppg_scl, fps, draw_fig_flag=False)
    single_cPPG_hma = hull_moving_average(filter_cppg_mdf,window_size=int(0.3*fps))

    # print(f'single_cPPG {single_cPPG}')
    # print(f'single_cPPG_ma {single_cPPG_ma}')
    # draw_figures(single_cppg_scl, filter_cppg_mdf, single_cPPG_hma, legend1='single_cPPG', legend2="filter_rPPG_btf",
    #              legend3='single_cPPG_hma')

    return single_cPPG_hma

# ECG processing
def Single_ECG_Processing(single_ECG, fps):

    # single_cPPG0 = single_cPPG
    single_ecg_scl = max_min_scaler(single_ECG)
    filter_ecg_mdf = signal_detrend_mdf(single_ecg_scl, fps, draw_fig=False)
    filter_ecg_btf = butterLowpassFilter(filter_ecg_mdf, lowcut=int(fps/3), fps=fps, order=4)

    # draw_figures(single_ecg_scl, filter_ecg_btf, filter_ecg_mdf, legend1="single_ecg_scl", legend2='filter_ecg_btf',
    #                  legend3='filter_ecg_mdf')

    return filter_ecg_btf



# # # --------------------------------- step 5: 数据存储 -----------------------------------
# 将数据存储为 H5 文件
def merging_label2h5(save_path,file_name,hr,cppg=None,cppg_fps=None,ecg=None,ecg_fps=None,bp=None,spo2=None):

    with h5py.File(save_path + '/' + file_name + ".h5", 'w') as f:
        f.create_dataset('hr', (1, 1), data=hr)
        if cppg is not None:
            f.create_dataset('cppg', (1, len(cppg)), data=cppg)
            f.create_dataset('cppg_fps', (1, 1), data=cppg_fps)
        if ecg is not None:
            f.create_dataset('ecg', (1, len(ecg)), data=ecg)
            f.create_dataset('ecg_fps', (1, 1), data=ecg_fps)
        if bp is not None:
            f.create_dataset('bp', (1, 2), data=bp)
        if spo2 is not None:
            f.create_dataset('spo2', (1, 1), data=spo2)

# —————————————————————————— 提取单位时长内的心率信息 ——————————————————————————
def Generate_HR_label_fn(HR, tp=60, tw=30):
    HR_arr = np.array(HR)
    if len(HR_arr) < int(tw * 0.6):
        HR_estmate = np.random.normal(loc=np.median(HR_arr), scale=3.0, size=(int(tw),))
        HR_estmate = moving_avg_3(HR_estmate, window_size=int(tw * 0.5))
    else:
        HR_estmate = HR_arr

    HR_estmate_ma_CI = Cubic_Interpolation(HR_estmate, fps=1,
                                                   time_points=int(tp * tw),
                                                   time_duration=tw)

    # draw_figures(HR_estmate, HR_estmate_ma,HR_estmate_ma_CI , legend1= 'HR_estmate',
    #               legend2= 'HR_estmate_ma', legend3= 'HR_estmate_ma_CI' )
    return HR_estmate_ma_CI

# —————————————————————————— 提取单位时长内的血氧信息 ——————————————————————————
def Generate_SPO2_label_fn(SPO2=98, tp=60, tw=30):
    SPO2_arr = np.array(SPO2)
    if len(SPO2_arr) < int(tw * 0.6):
        SPO2_estmate = np.random.normal(loc=np.median(SPO2_arr), scale=2.0, size=(int(tw),))
        SPO2_estmate = np.where(SPO2_estmate <= 100, SPO2_estmate, 100)
        SPO2_estmate = moving_avg_3(SPO2_estmate, window_size=int(tw * 0.5))
    else:
        SPO2_estmate = SPO2_arr

    SPO2_estmate_ma_CI = Cubic_Interpolation(SPO2_estmate, fps=1,
                                                   time_points=int(tp * tw),
                                                   time_duration=tw)
    return SPO2_estmate_ma_CI

# —————————————————————————— 提取单位时长内的收缩压信息 ——————————————————————————
def Generate_SBP_label_fn(SBP=115, HR=75, tp=60, tw=30):
    SBP_arr = np.array(SBP)
    if len(SBP_arr) < 4:
        SBP_estmate = np.random.normal(loc=np.median(SBP_arr) + int(20 * ((75 - HR) / HR)), scale=3.0, size=(int(tw),))
        SBP_estmate = moving_avg_3(SBP_estmate, window_size=int(tw * 0.5))
    else:
        SBP_estmate = SBP_arr

    SBP_estmate_ma_CI = Cubic_Interpolation(SBP_estmate, fps=1,
                                                   time_points=int(tp * tw),
                                                   time_duration=tw
                                            )
    return SBP_estmate_ma_CI

# —————————————————————————— 提取单位时长内的舒张压信息 ——————————————————————————
def Generate_DBP_label_fn(DBP=75, HR=75, tp=60, tw=30):
    DBP_arr = np.array(DBP)
    if len(DBP_arr) < 4:
        DBP_estmate = np.random.normal(loc=np.median(DBP_arr) + int(10 * ((75 - HR) / HR)), scale=2.0, size=(int(tw),))
        DBP_estmate = moving_avg_3(DBP_estmate, window_size=int(tw * 0.5))
    else:
        DBP_estmate = DBP_arr

    DBP_estmate_ma_CI = Cubic_Interpolation(DBP_estmate, fps=1,
                                                   time_points=int(tp * tw),
                                                   time_duration=tw)
    return DBP_estmate_ma_CI

def Clip_label_fn_V2(lables_df, label, time='None', tp=60, tw=30, ts=1, time_duration=None):

    # 无时间节点时传入time_duration
    if time != 'None':
        time_duration = int(lables_df[time].values[-1] - lables_df[time].values[0])
    elif time_duration != 'None':
        time_duration = time_duration
    else:
        print('Error: Clip_label_fn time_duration is missing !')
        print('lables_df',lables_df)
        print(f'time_duration {time_duration}')
        pdb.set_trace()

    fps = lables_df.shape[0] / time_duration

    # 每段信号时长为 tw = 30秒，间隔 ts = 1秒 切取一次
    clip_labels = []

    for i in range(0, int(time_duration - tw + 1), ts):

        temp_label_df= lables_df.iloc[int(i * fps):int((i + tw) * fps), :]

        # —————————————————————————— 提取单位时长内的心率信息 ——————————————————————————
        temp_label = temp_label_df[label]
        # print('temp_label_df',temp_label_df)

        # 去除预测的异常值
        label_median = np.median(temp_label)
        # print(f'temp_label_HR', len(temp_label_HR), temp_label_HR)
        # print('HR', HR)
        # pdb.set_trace()

        # 标签文件中的异常值
        temp_label_final = [i for i in temp_label if abs(i - label_median) / label_median <= 0.3]
        # print(f'temp_label_HR 剔除异常值后',  len(temp_label_HR_estmate), temp_label_HR_estmate)

        try:
            temp_label_final_CI = Cubic_Interpolation(temp_label_final, fps=0,
                                                           time_points=int(tp * tw),
                                                           time_duration=tw)
            # pdb.set_trace()

        except:
        # else:

            print(f"Warning:Clip_label_fn， {label} is random generated !")
            label_estmate = np.random.normal(loc=label_median, scale=2.0, size=(int(tw),))
            if label == 'SPO2':
                label_estmate = np.where(label_estmate <= 100, label_estmate, 100)
            label_estmate_ma = moving_avg_3(label_estmate, window_size=int(tw * 0.5))
            temp_label_final_CI = Cubic_Interpolation(label_estmate_ma, fps,
                                                           time_points=int(tp * tw),
                                                           time_duration=tw)

        # draw_2_figure(temp_label_HR, temp_label_HR_estmate_CI, legend1='temp_label_HR',
        #               legend2='temp_label_HR_estmate', )
    return temp_label_final_CI


def Clip_label_fn(lables_df, HR, SPO2, SBP, DBP, time='None', tp=60, tw=30, ts=1, time_duration=None):

    # 无时间节点时传入time_duration
    if time != 'None':
        time_duration = int(lables_df[time].values[-1] - lables_df[time].values[0])
    elif time_duration != 'None':
        time_duration = time_duration
    else:
        print('Error: Clip_label_fn time_duration is missing !')
        print('lables_df',lables_df)
        print(f'time_duration {time_duration}')
        pdb.set_trace()

    fps = lables_df.shape[0] / time_duration

    # 每段信号时长为 tw = 30秒，间隔 ts = 1秒 切取一次
    clip_labels = []

    for i in range(0, int(time_duration - tw + 1), ts):

        temp_label = lables_df.iloc[int(i * fps):int((i + tw) * fps), :]

        # —————————————————————————— 提取单位时长内的心率信息 ——————————————————————————
        temp_label_HR = temp_label[HR]
        # print('temp_label_HR',temp_label_HR)

        # 去除预测的异常值
        HR_median = np.median(temp_label_HR)
        # print(f'temp_label_HR', len(temp_label_HR), temp_label_HR)
        # print('HR', HR)
        # pdb.set_trace()

        # 标签文件中的异常值
        temp_label_HR_estmate = [i for i in temp_label_HR if abs(i - HR_median) / HR_median <= 0.3]
        # print(f'temp_label_HR 剔除异常值后',  len(temp_label_HR_estmate), temp_label_HR_estmate)

        try:
            temp_label_HR_estmate_CI = Cubic_Interpolation(temp_label_HR_estmate, fps,
                                                           time_points=int(tp * tw),
                                                           time_duration=tw)
            # pdb.set_trace()

        except:
        # else:
            print("Warning:Clip_label_fn， HR is random generated !")
            HR_estmate = np.random.normal(loc=HR_median, scale=3.0, size=(int(tw),))
            # print(HR_estmate)
            HR_estmate_ma = moving_avg_3(HR_estmate, window_size=int(tw * 0.5))
            temp_label_HR_estmate_CI = Cubic_Interpolation(HR_estmate_ma, fps,
                                                           time_points=int(tp * tw),
                                                           time_duration=tw)

        # draw_2_figure(temp_label_HR, temp_label_HR_estmate_CI, legend1='temp_label_HR',
        #               legend2='temp_label_HR_estmate', )

        # —————————————————————————— 提取单位时长内的血氧信息 ——————————————————————————
        if SPO2 != 'None':
            temp_label_SPO2 = temp_label[SPO2]
            # 去除预测的异常值
            temp_label_SPO2_median = np.median(temp_label_SPO2)
            temp_label_SPO2_estmate = [i for i in temp_label_SPO2 if abs(i - temp_label_SPO2_median) / temp_label_SPO2_median <= 0.2]
            temp_label_SPO2_estmate_CI = Cubic_Interpolation(temp_label_SPO2_estmate, fps,
                                                             time_points=int(tp * tw),
                                                             time_duration=time_duration)
            # draw_2_figure(HR_estmate,HR_estmate_CI,legend1='HR_estmate', legend2='HR_estmate_CI',)
        else:
            SPO2_estmate = np.random.normal(loc=97.0, scale=2.0, size=(int(tw),))
            # print(SPO2_estmate)
            SPO2_estmate = np.where(SPO2_estmate <= 100, SPO2_estmate, 100)
            # print(SPO2_estmate)
            SPO2_estmate_ma = moving_avg_3(SPO2_estmate, window_size=int(tw * 0.8))
            temp_label_SPO2_estmate_CI = Cubic_Interpolation(SPO2_estmate_ma, fps=tw, time_points=int(tp * tw))


        # —————————————————————————— 提取单位时长内的收缩压信息 ——————————————————————————
        if SBP != 'None':
            temp_label_SBP = temp_label[SBP]
            # 去除预测的异常值
            temp_label_SBP_median = np.median(temp_label_SBP)

            if np.std(temp_label_SBP) != 0:
                temp_label_SBP_estmate = [i for i in temp_label_SBP if abs(i - temp_label_SBP_median) / temp_label_SBP_median <= 0.3]
                temp_label_SBP_estmate_CI = Cubic_Interpolation(temp_label_SBP_estmate, fps,
                                                                time_points=int(tp * tw),
                                                                time_duration=tw)
            else:
                SBP_estmate = np.random.normal(loc=temp_label_SBP_median, scale=3.0, size=(int(tw),))
                SBP_estmate_ma = moving_avg_3(SBP_estmate, window_size=int(tw * 0.8))
                temp_label_SBP_estmate_CI = Cubic_Interpolation(SBP_estmate_ma, fps=tw, time_points=int(tp * tw))

            # draw_2_figure(HR_estmate,HR_estmate_CI,legend1='HR_estmate', legend2='HR_estmate_CI',)
        else:
            SBP_estmate = np.random.normal(loc=115.0 + int(20 * ((75 - HR_median) / HR_median)), scale=3.0, size=(int(tw),))
            SBP_estmate_ma = moving_avg_3(SBP_estmate, window_size=int(tw * 0.8))
            temp_label_SBP_estmate_CI = Cubic_Interpolation(SBP_estmate_ma, fps=tw, time_points=int(tp * tw))

        # —————————————————————————— 提取单位时长内的舒张压信息 ——————————————————————————
        if DBP != 'None':
            temp_label_DBP = temp_label[DBP]
            # 去除预测的异常值
            temp_label_DBP_median = np.median(temp_label_DBP)
            if np.std(temp_label_DBP) != 0:
                temp_label_DBP_estmate = [i for i in temp_label_DBP if
                                          abs(i - temp_label_DBP_median) / temp_label_DBP_median <= 0.3]

                temp_label_DBP_estmate_CI = Cubic_Interpolation(temp_label_DBP_estmate, fps,
                                                                time_points=int(tp * tw),
                                                                time_duration=tw)
            else:
                DBP_estmate = np.random.normal(loc=temp_label_DBP_median, scale=2.0, size=(int(tw),))
                DBP_estmate_ma = moving_avg_3(DBP_estmate, window_size=int(tw * 0.8))
                temp_label_DBP_estmate_CI = Cubic_Interpolation(DBP_estmate_ma, fps=tw, time_points=int(tp * tw))

            # draw_2_figure(HR_estmate,HR_estmate_CI,legend1='HR_estmate', legend2='HR_estmate_CI',)
        else:
            DBP_estmate = np.random.normal(loc=75.0 + int(10 * ((75 - HR_median) / HR_median)), scale=2.0, size=(int(tw),))
            DBP_estmate_ma = moving_avg_3(DBP_estmate, window_size=int(tw * 0.8))
            temp_label_DBP_estmate_CI = Cubic_Interpolation(DBP_estmate_ma, fps=tw, time_points=int(tp * tw))


        #  —————————————————————————— 调整单位时长内的标签信息，并数据保留三位小数 ——————————————————————————
        temp_label_HR_estmate_CI = [np.around(i,decimals=3 ) for i in temp_label_HR_estmate_CI]
        temp_label_SPO2_estmate_CI = [np.around(i, decimals=3) for i in temp_label_SPO2_estmate_CI]
        temp_label_SBP_estmate_CI = [np.around(i, decimals=3) for i in temp_label_SBP_estmate_CI]
        temp_label_DBP_estmate_CI = [np.around(i, decimals=3) for i in temp_label_DBP_estmate_CI]

        clip_labels.append([temp_label_HR_estmate_CI, temp_label_SPO2_estmate_CI, temp_label_SBP_estmate_CI, temp_label_DBP_estmate_CI])

        # draw_figures(HR_estmate_CI,temp_label_SPO2_estmate_CI,temp_label_SBP_estmate_CI,legend1='HR_estmate_CI',
        #              legend2='temp_label_SPO2_estmate_CI',legend3='temp_label_SBP_estmate_CI')

    clip_label_df = pd.DataFrame(clip_labels)
    return clip_label_df


def Clip_Signals_fn(signals, fps, tp=60, tw=30, ts=1, HR_estmate_flag=False):
    # draw_1_figure(signals,title='signals')

    signals = np.squeeze(signals)
    signals_num = signals.shape[0]
    # print(f'signals {signals_num}')

    time_duration = signals_num / fps
    # print(f'time_duration {time_duration}')
    # pdb.set_trace()

    # 每段信号时长为 tw = 30秒，间隔 ts = 1秒 切取一次
    segment_labels = []
    segment_features = []
    segment_signals = []
    # segment_signals_timepoint = []
    for i in range(0, int(time_duration - tw + 1), ts):
        temp_signal = signals[int(i * fps):int((i + tw) * fps)]

        # 数据归一化处理
        temp_segment_scalered = np.expand_dims(temp_signal, axis=1)
        temp_signal = max_min_scaler(temp_segment_scalered)
        temp_signal = np.squeeze(temp_signal)
        # print(f'temp_signal {temp_signal.shape}')

        # 计算心率
        HR_estmate_fft = fourier_analysis(temp_signal, fps=fps, draw_figures_flag=False)
        # 剔除标签异常的信号
        HR_estmate = find_peaks_analysis(temp_signal, fps=fps, draw_figures_flag=False)
        # 去除预测异常的信号
        HR_median = np.median(HR_estmate)
        # print(f'HR_estmate_fft {HR_estmate_fft}, HR{HR} ')
        # pdb.set_trace()

        # print(f'HR {HR}')
        if HR_median > 42 or HR_median < 180 and len(HR_estmate) > 20:
            temp_signal_CI = Cubic_Interpolation(temp_signal, fps, time_points=int(tp * tw))

            signal_features_list = Extract_Signal_Features(temp_signal_CI, fps=tp, time_points=int(tp * tw))

            segment_features.append(signal_features_list)

            filter_temp_signal_CI = butterLowpassFilter(temp_signal_CI, lowcut=int(tp / 3), fps=tp, order=6)

            segment_signals.append(list(filter_temp_signal_CI))

            # 根据分割信号计算标签信息
            if HR_estmate_flag:

                # 去除预测的异常值
                HR_estmate_ = [i for i in HR_estmate if abs(i-HR_median)/HR_median <= 0.3]
                # print(f'HR_estmate {len(HR_estmate)}, {HR_estmate}')
                # print(f'HR_estmate_ {len(HR_estmate_)}, {HR_estmate_}')

                HR_estmate_CI = Generate_HR_label_fn(HR=HR_estmate_, tp=tp, tw=tw)
                # SPO2_estmate_CI = Generate_SPO2_label_fn(SPO2=98, tp=tp, tw=tw)


                segment_labels.append(HR_estmate_CI)
                # draw_2_figure(HR_estmate, HR_estmate_CI, legend1='HR_estmate',legend2='HR_estmate_CI', )

        else:
            # 异常标签数据填充 0 值
            HR_estmate_check = find_peaks_analysis(temp_signal, fps=fps, draw_figures_flag=False)
            temp_signal_zero = np.zeros(int(tp * tw), )
            segment_signals.append(list(temp_signal_zero))
            segment_labels.append(list(temp_signal_zero))
            segment_features.append(list(temp_signal_zero))
            print(f'Warning,Clip_Signals_fn HR_median {HR_median}, Error in HR_estmate_check {HR_estmate_check}' )
            print(f'Warning,temp_signal_zero {len(temp_signal_zero)}, {temp_signal_zero}')
            # pdb.set_trace()

        # draw_figures(temp_signal_CI,temp_signal_CI,filter_temp_signal_CI,legend1='temp_signal',
        #              legend2='temp_signal_CI',legend3='filter_temp_signal_CI')
        # pdb.set_trace()

    segment_label_df = pd.DataFrame(segment_labels)

    segment_signal_df = pd.DataFrame(segment_signals)

    segment_feature_df = pd.DataFrame(segment_features)

    # print(f'segment_label_df {segment_label_df.shape}')
    # print(f'segment_signal_df {segment_signal_df.shape}')
    # print(f'segment_feature_df {segment_feature_df.shape}')
    # pdb.set_trace()

    if HR_estmate_flag:
        return segment_label_df, segment_feature_df, segment_signal_df
    else:
        return segment_feature_df, segment_signal_df


def dwt_fn(x,y,w=1,draw_fig_flag=False):

    dist, cost, acc, path = accelerated_dtw(x, y, dist='euclidean', warp=w)
    dist = np.around(dist,3)
    # Vizualize
    if draw_fig_flag:
        plt.imshow(cost.T, origin='lower', cmap=plt.cm.Reds, interpolation='nearest')
        plt.plot(path[0], path[1], '-o')  # relation
        plt.xticks(range(len(x)), x)
        plt.yticks(range(len(y)), y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('tight')
        plt.title('Minimum distance: {}, window widht: {}'.format(dist, w))
        plt.show()
    return dist

def clip_data_fn(data, fps, win_time, giveup_time=5):
    data = np.array(data)
    start_idx = int(fps*giveup_time)
    time_ = data.shape[0] / fps

    if int(win_time + giveup_time) <= time_:
        end_idx = int(fps * int(win_time + giveup_time))
    else:
        end_idx = int(fps * (time_ - giveup_time))

    data = data[start_idx: end_idx]
    return data

# 获取心率的中值
# def get_HR_label(HR_list):
#     ERROR = 100
#     HR = None
#     for i in HR_list:
#         error = abs(i - np.median(HR_list))
#         if error < ERROR:
#             HR = i
#             ERROR = error
#     return HR



def save_data_dict2h5(OutputFile,dict,win_time,fps):

    Roi_list = ['Roi_0', 'Roi_1', 'Roi_2', 'Roi_3', 'Roi_4', 'Roi_5', 'Roi_6', 'Roi_7', 'Roi_8']
    length = win_time * fps

    if not os.path.isfile(OutputFile):
        with h5py.File(OutputFile, 'a') as f:
            for i in Roi_list:
                bvp_roi = 'bvp_' + i
                ibvp_roi = 'ibvp_' + i
                f.create_dataset(bvp_roi,shape=(0,length), maxshape=(None, length), chunks=(100, length))
                f.create_dataset(ibvp_roi,shape=(0,length), maxshape=(None, length), chunks=(100, length))
            f.create_dataset('label', (0, 3), maxshape=(None, 3), dtype=int, chunks=(100, 3))
            f.create_dataset('ppg', (0, length), maxshape=(None, length), chunks=(100, length))
            f.create_dataset('subject_idx', (0, 1), maxshape=(None, 1), dtype=int, chunks=(100, 1))
            f.create_dataset('sample_id', (0, 1), maxshape=(None, 1), dtype=int, chunks=(100, 1))

    temp_length = 1
    # add data to .h5 file
    with h5py.File(OutputFile, "a") as f:
        BP_dataset = f['label']
        ppg_dataset = f['ppg']
        subject_dataset = f['subject_idx']
        sample_dataset = f['sample_id']

        DatasetCurrLength = BP_dataset.shape[0]
        DatasetNewLength = DatasetCurrLength + temp_length

        BP_dataset.resize(DatasetNewLength, axis=0)
        ppg_dataset.resize(DatasetNewLength, axis=0)
        subject_dataset.resize(DatasetNewLength, axis=0)
        sample_dataset.resize(DatasetNewLength, axis=0)

        BP_dataset[-temp_length:, :] = dict['label']
        ppg_dataset[-temp_length:, :] = dict['ppg']
        subject_dataset[-temp_length:, :] = dict['subject_idx']
        sample_dataset[-temp_length:, :] = dict['sample_id']

        for i in Roi_list:
            bvp_roi = 'bvp_' + i
            ibvp_roi = 'ibvp_' + i
            bvp_roi_dataset = f[bvp_roi]
            ibvp_roi_dataset = f[ibvp_roi]
            bvp_roi_dataset.resize(DatasetNewLength, axis=0)
            ibvp_roi_dataset.resize(DatasetNewLength, axis=0)
            # print(bvp_roi, len(dict[bvp_roi]))
            bvp_roi_dataset[-temp_length:,:] = dict[bvp_roi]
            ibvp_roi_dataset[-temp_length:,:] = dict[ibvp_roi]

def save_bestdata_dict2h5(OutputFile,dict,win_time,fps):

    length = win_time * fps
    if not os.path.isfile(OutputFile):
        with h5py.File(OutputFile, 'a') as f:
            f.create_dataset('bvp', shape=(0, length), maxshape=(None, length), chunks=(100, length))
            f.create_dataset('ibvp', shape=(0, length), maxshape=(None, length), chunks=(100, length))
            f.create_dataset('label', (0, 3), maxshape=(None, 3), dtype=int, chunks=(100, 3))
            f.create_dataset('ppg', (0, length), maxshape=(None, length), chunks=(100, length))
            f.create_dataset('subject_idx', (0, 1), maxshape=(None, 1), dtype=int, chunks=(100, 1))
            f.create_dataset('sample_id', (0, 1), maxshape=(None, 1), dtype=int, chunks=(100, 1))

    temp_length = 1
    # add data to .h5 file
    with h5py.File(OutputFile, "a") as f:
        BP_dataset = f['label']
        ppg_dataset = f['ppg']
        subject_dataset = f['subject_idx']
        sample_dataset = f['sample_id']
        bvp_roi_dataset = f['bvp']
        ibvp_roi_dataset = f['ibvp']

        DatasetCurrLength = BP_dataset.shape[0]
        DatasetNewLength = DatasetCurrLength + temp_length

        BP_dataset.resize(DatasetNewLength, axis=0)
        ppg_dataset.resize(DatasetNewLength, axis=0)
        subject_dataset.resize(DatasetNewLength, axis=0)
        sample_dataset.resize(DatasetNewLength, axis=0)
        bvp_roi_dataset.resize(DatasetNewLength, axis=0)
        ibvp_roi_dataset.resize(DatasetNewLength, axis=0)

        BP_dataset[-temp_length:, :] = dict['label']
        ppg_dataset[-temp_length:, :] = dict['ppg']
        subject_dataset[-temp_length:, :] = dict['subject_idx']
        sample_dataset[-temp_length:, :] = dict['sample_id']
        bvp_roi_dataset[-temp_length:, :] = dict['bvp']
        ibvp_roi_dataset[-temp_length:, :] = dict['ibvp']


## 存储7s,fps=125的rppg和cppg预测血压的训练文件，KeysViewHDF5 ['label', 'rppg', 'subject_idx']
def save_H5(rppg,rppg_ifft,cppg,bp,fps,OutputFile,subject_id,resample_flag=True):
    N_samp = min(len(rppg),len(cppg))
    win_time = 7
    new_fps = 125
    length = win_time*new_fps
    idx_start, idx_stop = CreateWindows(N_samp, win_time, step_time=3, fs=fps)
    # print('idx_start',idx_start)

    BP_list = []
    ppg_list = []
    rppg_list = []
    rppg_ifft_list = []

    for i in range(len(idx_start)):
        clip_cppg = cppg[idx_start[i]:idx_stop[i]]
        clip_rppg = rppg[idx_start[i]:idx_stop[i]]
        clip_rppg_ifft = rppg_ifft[idx_start[i]:idx_stop[i]]
        clip_bp = np.round(np.mean(bp[i:int(i + 7)],axis=0),2)
        BP_list.append(clip_bp)

        if resample_flag:
            clip_cppg = Cubic_Interpolation(clip_cppg, fps, time_points=new_fps)
            clip_rppg = Cubic_Interpolation(clip_rppg, fps, time_points=new_fps)
            clip_rppg_ifft = Cubic_Interpolation(clip_rppg_ifft, fps, time_points=new_fps)

        clip_cppg = mean_max_norm(clip_cppg)
        clip_rppg = mean_max_norm(clip_rppg)
        clip_rppg_ifft = mean_max_norm(clip_rppg_ifft)

        ppg_list.append(clip_cppg)
        rppg_list.append(clip_rppg)
        rppg_ifft_list.append(clip_rppg_ifft)

        # print('clip_label',clip_label)
        # print('clip_rppg_ifft', clip_rppg_ifft.shape)

        # draw_3_signals(clip_rppg, clip_rppg_ifft, clip_cppg, fps=125, legend1='rppg', legend2='rppg_ifft',legend3='cppg',
        #                title='save_H5',file_name=file_name + '_' + str(i), save_path=save_path, draw_fig_flag=True)

    BP_arr = np.array(BP_list)
    # print('BP_arr',BP_arr.shape)
    ppg_arr = np.array(ppg_list)
    rppg_arr = np.array(rppg_list)
    rppg_ifft_arr = np.array(rppg_ifft_list)
    temp_length = BP_arr.shape[0]

    if not os.path.isfile(OutputFile):
        with h5py.File(OutputFile, 'a') as f:
            f.create_dataset('label', (0, 2), maxshape=(None, 2), dtype=int, chunks=(100, 2))
            f.create_dataset('ppg', (0, length), maxshape=(None, length), chunks=(100, length))
            f.create_dataset('rppg', (0, length), maxshape=(None, length), chunks=(100, length))
            f.create_dataset('rppg_ifft', (0, length), maxshape=(None, length), chunks=(100, length))
            f.create_dataset('subject_idx', (0, 1), maxshape=(None, 1), dtype=int, chunks=(100, 1))

    if os.path.isfile(OutputFile):
        # add data to .h5 file
        with h5py.File(OutputFile, "a") as f:
            BP_dataset = f['label']
            ppg_dataset = f['ppg']
            rppg_dataset = f['rppg']
            rppg_ifft_dataset = f['rppg_ifft']
            subject_dataset = f['subject_idx']

            DatasetCurrLength = BP_dataset.shape[0]
            DatasetNewLength = DatasetCurrLength + temp_length
            BP_dataset.resize(DatasetNewLength, axis=0)
            ppg_dataset.resize(DatasetNewLength, axis=0)
            rppg_dataset.resize(DatasetNewLength, axis=0)
            rppg_ifft_dataset.resize(DatasetNewLength, axis=0)
            subject_dataset.resize(DatasetNewLength, axis=0)

            BP_dataset[-temp_length:, :] = BP_arr
            ppg_dataset[-temp_length:, :] = ppg_arr
            rppg_dataset[-temp_length:, :] = rppg_arr
            rppg_ifft_dataset[-temp_length:, :] = rppg_ifft_arr
            subject_dataset[-temp_length:, :] = subject_id * np.ones((temp_length, 1))

            # print(subject_dataset[-temp_length:, :].shape)
            # print(np.ones((temp_length, 1)).shape,np.ones((temp_length, 1)))
            # print(BP_arr.shape, BP_arr)
            # pdb.set_trace()

# # # --------------------------------- step 4: main -----------------------------------


# 截取中间段数据,舍弃第1秒 和 最后1秒的数据