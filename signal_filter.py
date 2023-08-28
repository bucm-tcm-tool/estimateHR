import os.path
import pdb
from scipy.signal import lfilter, find_peaks, medfilt
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

import pywt
# from obspy.signal.detrend import polynomial

# # # -------------------------------- step 2: 滤波器 -----------------------------------
# 滤波 1：小波变换滤波器
def CWT_filter(signal, draw_fig_flag=False):
    w = pywt.Wavelet('db8')  # 选用Daubechies8小波

    maxlev = pywt.dwt_max_level(len(signal), w.dec_len)
    # print("maximum level is " + str(maxlev))

    threshold = 0.08  # Threshold for filtering
    # Decompose into wavelet components, to the level selected:
    coeffs = pywt.wavedec(signal, 'db8', level=maxlev)  # 将信号进行小波分解

    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]), mode='soft', substitute=0)  # 将噪声滤波

    filter_sig = pywt.waverec(coeffs, 'db8')  # 将信号进行小波重构

    if draw_fig_flag:
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(signal)
        plt.xlabel('time (s)')
        plt.ylabel('microvolts (uV)')
        plt.title("Raw signal")
        plt.subplot(2, 1, 2)
        plt.plot(filter_sig)

        plt.xlabel('time (s)')
        plt.ylabel('microvolts (uV)')
        plt.title("De-noised signal using wavelet techniques")
        plt.tight_layout()
        plt.show()
        plt.close()

    return filter_sig


# 滤波 2：巴特沃斯带通滤波器
def bandpass_filter(data, fps, cutoff=(0.7, 3), order=2.0):
    nyq = 0.5 * fps
    low = float(cutoff[0]) / float(nyq)
    high = float(cutoff[1]) / float(nyq)
    b, a = signal.butter(order, [low, high], btype='bandpass')
    filter_sig = lfilter(b, a, data)
    return filter_sig


# 基于Heartpy的巴特沃斯带通滤波器
def hp_band_filter(data, fps, cutoff=(0.7, 3), order=6):
    low = cutoff[0]
    up = cutoff[1]
    filter_sig = hp.filter_signal(data, cutoff=[low, up], sample_rate=fps, order=order,
                                  filtertype='bandpass')
    return filter_sig


# ECG信号的范围为 [0.48, 40]
# 基于Heartpy的巴特沃斯高通滤波器
def hp_high_filter(data, fps, cutoff=0.48, order=6):
    filter_sig = hp.filter_signal(data, cutoff, sample_rate=fps, order=order,
                                  filtertype='highpass')
    return filter_sig


# 基于Heartpy的巴特沃斯低通滤波器
def hp_low_filter(data, fps, cutoff=40, order=6):
    filter_sig = hp.filter_signal(data, cutoff, sample_rate=fps, order=order,
                                  filtertype='lowpass')
    return filter_sig


def ppg_signal_preprocess(ppg_signal, ppg_fps, cutoff=(0.7, 4.0),flag='rppg',de_flag='mdf'):

    # 去趋势
    if de_flag=='mdf':
        ppg_signal_de = signal_detrend_mdf(ppg_signal, ppg_fps, draw_fig_flag=False)
    else:
        ppg_signal_de = signal.detrend(ppg_signal, type='constant',overwrite_data=True)  # 实测无效

    # 数据标准归一化
    ppg_signal_scl = standar_scaler(ppg_signal_de)

    # 平滑处理cppg会损失谐波信息，平滑处理rppg能降噪
    if flag == 'rppg':
        # 信号过滤
        ppg_signal_hp = hp.filter_signal(ppg_signal_scl, cutoff, sample_rate=ppg_fps, filtertype='bandpass')
        ppg_signal_final = hull_moving_average(ppg_signal_hp, window_size=5)

        # draw_3_signals(ppg_signal, ppg_signal_scl, ppg_signal_final, fps=30, legend1='original signal',
        #                legend2='signal standar_scaler', legend3='signal smooth')

    elif flag == 'ppg':
        ppg_signal_final = smooth(ppg_signal_scl,3)
        # draw_3_signals(ppg_signal, ppg_signal_scl, ppg_signal_final, fps=30, legend1='original signal',
        #                legend2='signal standar_scaler', legend3='signal smooth')
    else:
        print(f'Warning: ppg_signal_preprocess not defined the type of ppg/rppg')
        ppg_signal_final = None
        pdb.set_trace()

    # draw_3_signals(ppg_signal, ppg_signal_de, ppg_signal_final, fps=30, legend1='original signal',
    #                legend2='signal detrend', legend3='signal filter')

    # 评价波形相似
    # dwt_fn(ppg_signal,ppg_signal_hp,w=1,draw_fig_flag=True)

    return ppg_signal_final


def ecg_signal_preprocess(ecg_signal, ecg_fps):
    # 信号过滤
    ecg_signal = hp.filter_signal(ecg_signal, cutoff=0.48, sample_rate=ecg_fps, order=6,
                                  filtertype='highpass')
    ecg_signal_hp = hp.filter_signal(ecg_signal, cutoff=40, sample_rate=ecg_fps, order=6,
                                     filtertype='lowpass')
    ecg_signal_cwt = CWT_filter(ecg_signal_hp, draw_fig_flag=False)
    # 标准化
    ecg_signal_scl = standar_scaler(ecg_signal_cwt)

    # draw_3_signals(ecg_signal, ecg_signal_cwt, ecg_signal_scl, legend1='ecg_signal', legend2='ecg_signal_cwt',
    #                legend3='ecg_signal_scl', title='ecg_signal_preprocess',)

    return ecg_signal_scl


# 滤波 3：fft的信号过滤器
def FFT_filter(signal, fps, fig_name='Figure', dataset=None, save_path=None,draw_fig_flag=False):

    # 数据标准化,及归一化
    signal = mean_max_norm(signal)
    signal = standar_scaler(signal)
    # 傅里叶变换
    signal_fft = np.fft.fft(signal)
    power = np.abs(signal_fft) ** 2
    freqs = np.fft.fftfreq(signal_fft.size, 1 / fps)  # 得到分解波的频率序列
    half_power = power[0 <= freqs]
    half_freqs = freqs[0 <= freqs]

    win = int(len(half_freqs) / (12 * np.max(half_freqs)) + 1)
    # print('win', win)

    # 去除异常噪声，增强心率信号
    # 剔除[0.7,5.0]之外的信号频谱
    signal_fft[abs(freqs) < 0.7] = 0
    signal_fft[abs(freqs) > 5.0] = 0

    # 根据原始信号的信噪比进行信号增强
    fft_snr_ori = SignaltoNoiseRatio_rPPG(signal, fps, draw_fig_flag=draw_fig_flag)
    power_max_idx_def = np.argmax(half_power)

    if fft_snr_ori < 0.7 and dataset == 'MSSD':
        # print(f'FFT Detect Warning: The Original SNR is {fft_snr_ori}.')
        power_max_index = get_power_maxidx(signal, fps)

    else:
        power_max_index = power_max_idx_def
    # print('power_max_idx_def', power_max_idx_def, 'get_power_maxidx', power_max_index)

    signal_fft_c0 = enhance_signal_fft(signal, fps, power_max_index, item=1)

    # 第一次逆变换信号
    signal_fft_c1 = fill_zero_nainf(signal_fft_c0)
    ifft_signal = np.fft.ifft(signal_fft_c1)
    ifft_signal_1st = ifft_signal
    # 第一次逆变换信号的SNR
    ifft_snr_1st = SignaltoNoiseRatio_rPPG(ifft_signal, fps, power_max_index, draw_fig_flag=draw_fig_flag)
    # print(f'ifft_snr_1st {ifft_snr_1st}')

    if draw_fig_flag:
        draw_2_signals(signal,ifft_signal_1st,fps=fps,
                       legend1='original signal snr ' + str(np.round(fft_snr_ori, 2)),
                       legend2='ifft_signal_1st snr ' + str(np.round(ifft_snr_1st, 2)),
                       title='1st ifft_signal'
                       )

    # 根据逆变换后信号的信噪比进行信号增强
    if ifft_snr_1st < 0.7:
        # 第二次逆变换信号
        signal_fft_c2 = enhance_signal_fft(ifft_signal, fps, power_max_index, item=1)
        signal_fft_c2 = fill_zero_nainf(signal_fft_c2)
        ifft_signal = np.fft.ifft(signal_fft_c2)
        ifft_signal_2nd = ifft_signal
        # 第二次逆变换信号的SNR
        ifft_snr_2nd = SignaltoNoiseRatio_rPPG(ifft_signal, fps, power_max_index, draw_fig_flag=draw_fig_flag)
        # print(f'ifft_snr_2nd {ifft_snr_2nd}')

        if draw_fig_flag:
            draw_3_signals(signal,ifft_signal_1st,ifft_signal_2nd,fps=fps,
                           legend1='original signal snr ' + str(np.round(fft_snr_ori,2)),
                           legend2='ifft_signal_1st snr ' + str(np.round(ifft_snr_1st,2)),
                           legend3='ifft_signal_2nd snr ' + str(np.round(ifft_snr_2nd,2)),
                           title='2nd ifft_signal'
                           )

        if ifft_snr_2nd < 0.7:
            # 第三次逆变换信号
            signal_fft_c3 = enhance_signal_fft(ifft_signal, fps, power_max_index, item=2)
            signal_fft_c3 = fill_zero_nainf(signal_fft_c3)
            ifft_signal = np.fft.ifft(signal_fft_c3)
            ifft_signal_3rd = ifft_signal

            # 第三次逆变换信号的SNR
            ifft_snr_3rd = SignaltoNoiseRatio_rPPG(ifft_signal, fps, power_max_index, draw_fig_flag=draw_fig_flag)
            # print(f'ifft_snr_3rd {ifft_snr_3rd}')

            if draw_fig_flag:
                draw_3_signals(ifft_signal_1st, ifft_signal_2nd, ifft_signal_3rd, fps=fps,
                               legend1='ifft_signal_1st snr ' + str(np.round(ifft_snr_1st, 2)),
                               legend2='ifft_signal_2nd snr ' + str(np.round(ifft_snr_2nd, 2)),
                               legend3='ifft_signal_3rd snr ' + str(np.round(ifft_snr_3rd, 2)),
                               title='3nd ifft_signal'
                               )


    # 数据标准化,及归一化
    ifft_signal = mean_max_norm(ifft_signal)
    ifft_signal = standar_scaler(ifft_signal)
    ifft_power_max_index = get_power_maxidx(ifft_signal, fps)
    ifft_snr = SignaltoNoiseRatio_rPPG(ifft_signal, fps)
    # pdb.set_trace()

    # # 可视化
    # draw_fig_flag = True
    if draw_fig_flag:
        # 逆变换后信号的频谱
        signal_ifft_fft = np.fft.fft(ifft_signal)
        power_ifft = np.abs(signal_ifft_fft) ** 2
        freqs_ifft = np.fft.fftfreq(signal_ifft_fft.size, 1 / fps)
        half_freqs_ifft = freqs_ifft[0 <= freqs_ifft]
        half_power_ifft = power_ifft[0 <= freqs_ifft]

        hr_freqs = half_freqs[power_max_idx_def]
        hr_estimate = np.around(hr_freqs * 60, decimals=2)
        print(f'original signal power_max_idx_def {power_max_idx_def}, signal {len(signal)}, ',
              f'hr_freqs {hr_freqs}, hr_estimate {hr_estimate}')

        hr_ifft_freqs = half_freqs_ifft[ifft_power_max_index]
        hr_ifft_estimate = np.round(hr_ifft_freqs * 60, decimals=2)
        print(f'ifft signal ifft_power_max_index,{ifft_power_max_index}, ifft_signal {len(ifft_signal)}, ',
              f'hr_ifft_freqs, {hr_ifft_freqs}, hr_ifft_estimate, {hr_ifft_estimate}')

        freq_main_region = half_freqs[int(power_max_idx_def - win):int(power_max_idx_def + win + 1)]
        freq_sub_region = half_freqs[int(power_max_idx_def - win) * 2:int(power_max_idx_def + win + 1) * 2]
        power_main_region = half_power[int(power_max_idx_def - win):int(power_max_idx_def + win + 1)]
        power_sub_region = half_power[int(power_max_idx_def - win) * 2:int(power_max_idx_def + win + 1) * 2]

        freq_main_region_ifft = half_freqs_ifft[int(ifft_power_max_index - win):int(ifft_power_max_index + win + 1)]
        freq_sub_region_ifft = half_freqs_ifft[int(ifft_power_max_index - win) * 2:int(ifft_power_max_index + win + 1) * 2]
        power_main_region_ifft = half_power_ifft[int(ifft_power_max_index - win):int(ifft_power_max_index + win + 1)]
        power_sub_region_ifft = half_power_ifft[int(ifft_power_max_index - win) * 2:int(ifft_power_max_index + win + 1) * 2]

        # 比较原始和降噪后的信号与频谱
        plt.figure('FFT_filter', figsize=(9, 6))
        plt.subplot(411)
        plt.plot(signal, label='HR ' + str(hr_estimate))
        plt.legend(loc='upper right')

        plt.subplot(412)
        plt.plot(half_freqs[half_freqs < 5], half_power[half_freqs < 5],)
        plt.vlines(hr_freqs, -1, int(np.max(half_power) * 1.2), colors='r', linestyles='dashed')
        plt.scatter(half_freqs[power_max_idx_def], half_power[power_max_idx_def], alpha=0.5,
                   label='hr_freqs ' + str(np.round(hr_freqs, 2)))
        plt.plot(freq_main_region, power_main_region, color='r', label='snr ' + str(np.round(fft_snr_ori,2)))
        plt.plot(freq_sub_region, power_sub_region, color='r')
        plt.legend(loc='upper right')

        plt.subplot(413)
        plt.plot(ifft_signal, color='g', label='HR_ifft  ' + str(hr_ifft_estimate))
        plt.legend(loc='upper right')

        plt.subplot(414)
        plt.plot(half_freqs_ifft[half_freqs_ifft < 5], half_power_ifft[half_freqs_ifft < 5], color='g')
        plt.vlines(hr_ifft_freqs, -1, int(np.max(half_power_ifft) * 1.2), colors='r', linestyles='dashed')
        plt.scatter(half_freqs_ifft[ifft_power_max_index], half_power_ifft[ifft_power_max_index], alpha=0.5,
                   label='hr_freqs ' + str(np.round(hr_ifft_freqs, 2)))
        plt.plot(freq_main_region_ifft, power_main_region_ifft, color='r', label='ifft_snr ' + str(np.round(ifft_snr,2)))
        plt.plot(freq_sub_region_ifft, power_sub_region_ifft, color='r')
        plt.legend(loc='upper right')

        if draw_fig_flag:
            plt.show()
        elif save_path != None:
            fig_path = save_path + '/Plot_Figs/'
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)
            plt.savefig(fig_path + fig_name + '.png')

        plt.close()

    # 逆变换后的数据为复数，无法进行后续处理，取实数部分会导致信息损失，预测值偏小
    ifft_signal = np.real(ifft_signal)
    return ifft_signal
# # 滤波 3：fft的信号过滤器
# def FFT_filter(signal, fps, method='fft', fig_name='Figure', dataset=None, save_path=None,draw_fig_flag=False):
#     # 基于stft计算信噪比
#     if method=='stft':
#         nperseg = int(min(len(signal) * 0.5, fps * 15))
#         # # print(f'nperseg {nperseg}')
#         freqs, power, _ = get_stft_freq_power(signal, fps, method='median', return_onesided=False, nperseg=nperseg)
#
#     # 基于fft计算信噪比
#     elif method == 'fft':
#         freqs, power, _ = get_fft_freq_power(signal, fps, return_onesided=False)
#
#     elif method == 'welch':
#         # # welch傅里叶变换
#         nperseg = int(min(len(signal) * 0.5, fps * 15))
#         # nperseg = len(signal)
#         freqs, power = welch(signal, fps, 'flattop', nperseg=nperseg, average='median', return_onesided=False)
#
#     else:
#         freqs, power = None,None
#         print('Warning: FFT_filter method does not defined!')
#         pdb.set_trace()
#
#
#     half_power = power[0 <= freqs]
#     half_freqs = freqs[0 <= freqs]
#
#     win = int(len(half_freqs) / (12 * np.max(half_freqs)) + 1)
#     # print('win', win)
#
#     # 去除异常噪声，增强心率信号
#     # 剔除[0.7,5.0]之外的信号频谱
#     power[abs(freqs) < 0.7] = 0
#     power[abs(freqs) > 5.0] = 0
#
#     # 根据原始信号的信噪比进行信号增强
#     fft_snr_ori = SignaltoNoiseRatio_rPPG(signal, fps, draw_fig_flag=draw_fig_flag)
#     power_max_idx_def = np.argmax(half_power)
#
#     if fft_snr_ori < 0.7 and dataset == 'MSSD':
#         print(f'FFT Detect Warning: The Original SNR is {fft_snr_ori}.')
#         power_max_index = get_power_maxidx(signal, fps)
#         snr = SignaltoNoiseRatio_rPPG(signal, fps, power_max_index=power_max_index, draw_fig_flag=draw_fig_flag)
#     else:
#         power_max_index = power_max_idx_def
#         snr = fft_snr_ori
#     print('power_max_idx_def', power_max_idx_def, 'get_power_maxidx', power_max_index)
#
#     signal_fft_c0 = enhance_signal_fft(freqs, power, power_max_index, snr, item=1)
#     print('signal_fft_c0', len(signal_fft_c0), signal_fft_c0)
#     # 第一次逆变换信号
#     signal_fft_c1 = fill_zero_nainf(signal_fft_c0)
#     ifft_signal = np.fft.ifft(signal_fft_c1)
#     ifft_signal_1st = ifft_signal
#     # 第一次逆变换信号的SNR
#     ifft_snr_1st = SignaltoNoiseRatio_rPPG(ifft_signal, fps, power_max_index, draw_fig_flag=draw_fig_flag)
#     print(f'ifft_snr_1st {ifft_snr_1st}')
#
#     if draw_fig_flag:
#         draw_2_signals(signal,ifft_signal_1st,fps=fps,
#                        legend1='original signal snr ' + str(np.round(fft_snr_ori, 2)),
#                        legend2='ifft_signal_1st snr ' + str(np.round(ifft_snr_1st, 2)),
#                        title='1st ifft_signal'
#                        )
#
#     # 根据逆变换后信号的信噪比进行信号增强
#     if ifft_snr_1st < 0.7:
#         # 第二次逆变换信号
#         signal_fft_c2 = enhance_signal_fft(ifft_signal, fps, power_max_index, item=1)
#         signal_fft_c2 = fill_zero_nainf(signal_fft_c2)
#         ifft_signal = np.fft.ifft(signal_fft_c2)
#         ifft_signal_2nd = ifft_signal
#         # 第二次逆变换信号的SNR
#         ifft_snr_2nd = SignaltoNoiseRatio_rPPG(ifft_signal, fps, power_max_index, draw_fig_flag=draw_fig_flag)
#         # print(f'ifft_snr_2nd {ifft_snr_2nd}')
#
#         if draw_fig_flag:
#             draw_3_signals(signal,ifft_signal_1st,ifft_signal_2nd,fps=fps,
#                            legend1='original signal snr ' + str(np.round(fft_snr_ori,2)),
#                            legend2='ifft_signal_1st snr ' + str(np.round(ifft_snr_1st,2)),
#                            legend3='ifft_signal_2nd snr ' + str(np.round(ifft_snr_2nd,2)),
#                            title='2nd ifft_signal'
#                            )
#
#         if ifft_snr_2nd < 0.7:
#             # 第三次逆变换信号
#             signal_fft_c3 = enhance_signal_fft(ifft_signal, fps, power_max_index, item=2)
#             signal_fft_c3 = fill_zero_nainf(signal_fft_c3)
#             ifft_signal = np.fft.ifft(signal_fft_c3)
#             ifft_signal_3rd = ifft_signal
#
#             # 第三次逆变换信号的SNR
#             ifft_snr_3rd = SignaltoNoiseRatio_rPPG(ifft_signal, fps, power_max_index, draw_fig_flag=draw_fig_flag)
#             # print(f'ifft_snr_3rd {ifft_snr_3rd}')
#
#             if draw_fig_flag:
#                 draw_3_signals(ifft_signal_1st, ifft_signal_2nd, ifft_signal_3rd, fps=fps,
#                                legend1='ifft_signal_1st snr ' + str(np.round(ifft_snr_1st, 2)),
#                                legend2='ifft_signal_2nd snr ' + str(np.round(ifft_snr_2nd, 2)),
#                                legend3='ifft_signal_3rd snr ' + str(np.round(ifft_snr_3rd, 2)),
#                                title='3nd ifft_signal'
#                                )
#
#
#     # 数据标准化,及归一化
#     ifft_signal = mean_max_norm(ifft_signal)
#     ifft_signal = standar_scaler(ifft_signal)
#     ifft_power_max_index = get_power_maxidx(ifft_signal, fps)
#     ifft_snr = SignaltoNoiseRatio_rPPG(ifft_signal, fps)
#     pdb.set_trace()
#
#     # # 可视化
#     # draw_fig_flag = True
#     if draw_fig_flag:
#         # 逆变换后信号的频谱
#         signal_ifft_fft = np.fft.fft(ifft_signal)
#         power_ifft = np.abs(signal_ifft_fft) ** 2
#         freqs_ifft = np.fft.fftfreq(signal_ifft_fft.size, 1 / fps)
#         half_freqs_ifft = freqs_ifft[0 <= freqs_ifft]
#         half_power_ifft = power_ifft[0 <= freqs_ifft]
#
#         hr_freqs = half_freqs[power_max_idx_def]
#         hr_estimate = np.around(hr_freqs * 60, decimals=2)
#         print(f'original signal power_max_idx_def {power_max_idx_def}, signal {len(signal)}, ',
#               f'hr_freqs {hr_freqs}, hr_estimate {hr_estimate}')
#
#         hr_ifft_freqs = half_freqs_ifft[ifft_power_max_index]
#         hr_ifft_estimate = np.round(hr_ifft_freqs * 60, decimals=2)
#         print(f'ifft signal ifft_power_max_index,{ifft_power_max_index}, ifft_signal {len(ifft_signal)}, ',
#               f'hr_ifft_freqs, {hr_ifft_freqs}, hr_ifft_estimate, {hr_ifft_estimate}')
#
#         freq_main_region = half_freqs[int(power_max_idx_def - win):int(power_max_idx_def + win + 1)]
#         freq_sub_region = half_freqs[int(power_max_idx_def - win) * 2:int(power_max_idx_def + win + 1) * 2]
#         power_main_region = half_power[int(power_max_idx_def - win):int(power_max_idx_def + win + 1)]
#         power_sub_region = half_power[int(power_max_idx_def - win) * 2:int(power_max_idx_def + win + 1) * 2]
#
#         freq_main_region_ifft = half_freqs_ifft[int(ifft_power_max_index - win):int(ifft_power_max_index + win + 1)]
#         freq_sub_region_ifft = half_freqs_ifft[int(ifft_power_max_index - win) * 2:int(ifft_power_max_index + win + 1) * 2]
#         power_main_region_ifft = half_power_ifft[int(ifft_power_max_index - win):int(ifft_power_max_index + win + 1)]
#         power_sub_region_ifft = half_power_ifft[int(ifft_power_max_index - win) * 2:int(ifft_power_max_index + win + 1) * 2]
#
#         # 比较原始和降噪后的信号与频谱
#         plt.figure('FFT_filter', figsize=(9, 6))
#         plt.subplot(411)
#         plt.plot(signal, label='HR ' + str(hr_estimate))
#         plt.legend(loc='upper right')
#
#         plt.subplot(412)
#         plt.plot(half_freqs[half_freqs < 5], half_power[half_freqs < 5],)
#         plt.vlines(hr_freqs, -1, int(np.max(half_power) * 1.2), colors='r', linestyles='dashed')
#         plt.scatter(half_freqs[power_max_idx_def], half_power[power_max_idx_def], alpha=0.5,
#                    label='hr_freqs ' + str(np.round(hr_freqs, 2)))
#         plt.plot(freq_main_region, power_main_region, color='r', label='snr ' + str(np.round(fft_snr_ori,2)))
#         plt.plot(freq_sub_region, power_sub_region, color='r')
#         plt.legend(loc='upper right')
#
#         plt.subplot(413)
#         plt.plot(ifft_signal, color='g', label='HR_ifft  ' + str(hr_ifft_estimate))
#         plt.legend(loc='upper right')
#
#         plt.subplot(414)
#         plt.plot(half_freqs_ifft[half_freqs_ifft < 5], half_power_ifft[half_freqs_ifft < 5], color='g')
#         plt.vlines(hr_ifft_freqs, -1, int(np.max(half_power_ifft) * 1.2), colors='r', linestyles='dashed')
#         plt.scatter(half_freqs_ifft[ifft_power_max_index], half_power_ifft[ifft_power_max_index], alpha=0.5,
#                    label='hr_freqs ' + str(np.round(hr_ifft_freqs, 2)))
#         plt.plot(freq_main_region_ifft, power_main_region_ifft, color='r', label='ifft_snr ' + str(np.round(ifft_snr,2)))
#         plt.plot(freq_sub_region_ifft, power_sub_region_ifft, color='r')
#         plt.legend(loc='upper right')
#
#         if draw_fig_flag:
#             plt.show()
#         elif save_path != None:
#             fig_path = save_path + '/Plot_Figs/'
#             if not os.path.exists(fig_path):
#                 os.makedirs(fig_path)
#             plt.savefig(fig_path + fig_name + '.png')
#
#         plt.close()
#
#     # 逆变换后的数据为复数，无法进行后续处理，取实数部分会导致信息损失，预测值偏小
#     ifft_signal = np.real(ifft_signal)
#     return ifft_signal

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


# # 小波变换去趋势
# def signal_detrend_cwt(signal, fps, draw_fig_flag=False):
#     # 设置小波变换的尺度范围，选择小波基
#     wavename = 'cgau8'
#     signal = np.squeeze(signal)
#     totalscal = signal.shape[0]
#     time_duration = totalscal / fps
#     t = np.linspace(0, time_duration, signal.shape[0])
#     fc = pywt.central_frequency(wavename)
#     cparam = 2 * fc * totalscal
#     scales = cparam / np.arange(totalscal, 1, -1)
#     [cwtcoff, frequencies] = pywt.cwt(signal, scales, wavename)  # 连续小波变换的返回值是时频图和频率
#
#     # 去除baseline
#     for i in range(totalscal - 1):
#         baseline = np.mean(cwtcoff[i][0:int(totalscal * 0.2)])  # 这里选了 20% 时间范围内的功率均值作为 baseline
#         for j in range(totalscal):
#             cwtcoff[i][j] = cwtcoff[i][j] - baseline  # 在每个频率上，原始功率减去对应的 baseline 值
#
#     # 绘制时频图
#     if draw_fig_flag:
#         plt.figure(figsize=(8, 4))
#         plt.contourf(t, frequencies, abs(cwtcoff))
#         plt.ylabel('frequency')
#         plt.xlabel('time(second)')
#         plt.colorbar()
#         plt.show()
#
#     return t, frequencies
