import matplotlib.pyplot as plt
from tools import *
from get_bvp_signal import get_bvp_signal
from tqdm import tqdm
import pywt
import h5py as h5



# # # --------------------------------- step3: 心率估计 -----------------------------------
# 方法1：峰值检测
def ibt_est_hr(signal, fps, htr=0.8, draw_fig_flag=False):

    # # 检测主波方向
    if fps >=75:
        signal = standar_scaler(signal)
        height_top10 = abs(np.mean(np.sort(signal)[-int(len(signal)/fps*3):-int(len(signal)/fps*0.25)])) # 适用于高帧率，ecg
        height_bottom10 = abs(np.mean(np.sort(signal)[int(len(signal)/fps*0.25):int(len(signal)/fps*3)]))
        height_diff = height_top10 - height_bottom10
        # height_diff = 0
        if height_diff >= 0:
            peaks, properties = find_peaks(signal, height=height_top10 * htr, distance=int(fps * 0.38))
        else:
            peaks, properties = find_peaks(-signal, height=height_bottom10 * htr, distance=int(fps * 0.38))
            # draw_fig_flag = True
    else:
        # 数据归一化 [0, 1],由于标准归一化会影响pkvc的计算
        signal = min_max_norm(signal)
        height_top10 = abs(np.mean(np.sort(signal)[-int(len(signal) * 0.25):-int(len(signal) * 0.05)]))
        peaks, properties = find_peaks(signal, height=height_top10 * htr, distance=int(fps * 0.38))


    # 根据峰值间距计算心率
    peaks_diff = np.diff(peaks)
    IBT = peaks_diff / fps

    # 评价心率变异性
    AVPP = np.round(np.mean(IBT), 3)
    SDPP = np.round(np.std(IBT), 3)
    PPVC = np.round(SDPP/AVPP, 3)
    IBT_feature = [AVPP, SDPP, PPVC]


    # 计算心率值
    HR_list = np.round(60 / IBT, 3)
    HR_med = np.round(np.median(HR_list), 2)
    HR_mean = np.round(np.mean(HR_list), 2)
    HR_ibt_pred = [HR_med,HR_mean]

    # 根据峰值高度的变异系数判断数据质量
    Peak_height = properties['peak_heights']
    Peak_height = min_max_norm(Peak_height)
    pkvc = np.std(Peak_height)/np.mean(Peak_height)  # Peak_height_variable_coefficient
    pkvc = np.round(pkvc, 3)

    ret = 1
    if min(PPVC, pkvc) > 0.2:

        ret = 0
        # draw_fig_flag = True

    if np.min(HR_list) < 30 or np.max(HR_list) > 180:

        ret = 0


    # 查看数据
    # draw_fig_flag = True
    if draw_fig_flag:
        x = np.arange(len(signal))
        plt.figure('Interbeats_est_hr', figsize=(9, 6))
        plt.plot(x, signal, label='HR_med ' + str(np.round(HR_med,2)))
        plt.scatter(peaks, signal[peaks], color='r', alpha=0.5,label='phvc = ' + str(np.round(pkvc,3)))
        plt.legend(loc='upper right')
        plt.show()
        plt.close()

    return HR_ibt_pred, IBT_feature, ret

# 方法2：傅里叶分析
# 基于fft预测心率
# 基于快速傅里叶变换预测心率
def fft_est_hr(signal, fps, draw_fig_flag=False):

    half_freqs, half_power, win = get_fft_freq_power(signal, fps)
    # 剔除[0.5,5.0]之外的信号频谱
    half_power[half_freqs < 0.7] = 0
    half_power[half_freqs > 3.0] = 0

    # 基于最大频谱值确定心率的频谱
    power_max_default = np.argmax(half_power)
    hr_freqs_default = half_freqs[power_max_default]
    hr_est_default = np.around(hr_freqs_default * 60, decimals=2)
    # print(f'power_max_default {power_max_default}, hr_freqs_default {hr_freqs_default}, hr_est_default {hr_est_default}')

    # 基于最大信噪比确定心率的频谱
    power_max_snr = get_power_maxidx(signal, fps, method='fft')
    hr_freqs_snr = half_freqs[power_max_snr]
    hr_est_snr = np.around(hr_freqs_snr * 60, decimals=2)


    # draw_fig_flag = True
    if draw_fig_flag:
        HR_fft_list = [hr_est_default, hr_est_snr]
        print(f'hr_est_default {hr_est_default},  hr_est_snr {hr_est_snr}')
        fft_snr_default = SignaltoNoiseRatio_rPPG(signal, fps, power_max_index=power_max_default)
        power_main_region_default = half_power[int(power_max_default - win):int(power_max_default + win + 1)]
        freq_main_reqion_default = half_freqs[int(power_max_default - win):int(power_max_default + win + 1)]


        fft_snr_ = SignaltoNoiseRatio_rPPG(signal, fps, power_max_index=power_max_snr)
        power_main_region_ = half_power[int(power_max_snr - win):int(power_max_snr + win + 1)]
        freq_main_reqion_ = half_freqs[int(power_max_snr - win):int(power_max_snr + win + 1)]

        # 比较原始和降噪后的信号与频谱
        plt.figure('fft_est_hr',figsize=(9,6))
        plt.subplot(311)
        plt.plot(signal, label='signal HR ' + str(HR_fft_list))
        plt.legend(loc='upper right')

        plt.subplot(312)
        plt.plot(half_freqs[half_freqs < 5], half_power[half_freqs < 5], label='SNR ' + str(np.round(fft_snr_default, 2)))
        plt.vlines(hr_freqs_default, 0, int(np.max(half_power) * 1.2), colors='r', linestyles='dashed',
                                  label='hr_freqs ' + str(np.round(hr_freqs_default, 2)))
        plt.plot(freq_main_reqion_default, power_main_region_default, 'r')

        plt.scatter(half_freqs[power_max_default], half_power[power_max_default], alpha=0.5)
        plt.legend(loc='upper right')

        plt.subplot(313)
        plt.plot(half_freqs[half_freqs < 5], half_power[half_freqs < 5],
                 label='SNR ' + str(np.round(fft_snr_, 2)))
        plt.vlines(hr_freqs_snr, 0, int(np.max(half_power) * 1.2), colors='r', linestyles='dashed',
                   label='hr_freqs ' + str(np.round(hr_freqs_snr, 2)))
        plt.plot(freq_main_reqion_, power_main_region_, 'r')

        plt.scatter(half_freqs[power_max_snr], half_power[power_max_snr], alpha=0.5)
        plt.legend(loc='upper right')

        plt.show()
        plt.close()

    return hr_est_default, hr_est_snr

# 基于短时傅里叶变换预测心率
def stft_est_hr(signal, fps, draw_fig_flag=False):

    nperseg = int(min(len(signal) * 0.5, fps * 15))

    half_freqs, half_powers_med, _ = get_stft_freq_power(signal, fps, method='median', return_onesided=True, nperseg=nperseg)
    half_freqs, half_powers_mean, _ = get_stft_freq_power(signal, fps, method='mean', return_onesided=True,
                                                         nperseg=nperseg)


    signal_sfft_maxidx_median =np.argmax(half_powers_med)
    HR_med = np.around(abs(half_freqs[np.argmax(half_powers_med)]) * 60, 2)


    signal_sfft_maxidx_mean = np.argmax(half_powers_mean)
    HR_mean = np.around(abs(half_freqs[np.argmax(half_powers_mean)]) * 60, 2)


    if draw_fig_flag:
        print('stft_est_hr HR_med', HR_med, 'HR_mean ', HR_mean)

        plt.figure('stft_est_hr', figsize=(9, 6))
        plt.subplot(311)
        plt.plot(signal, label='signal HR ' + str([HR_med,HR_mean]))
        plt.legend(loc='upper right')

        plt.subplot((312))
        plt.plot(half_freqs, half_powers_med)
        plt.vlines(half_freqs[signal_sfft_maxidx_median], 0, np.max(half_powers_med) * 1.2, colors='r', linestyles='--',
                   label='HR_med ' + str(HR_med))
        plt.legend(loc='upper right')

        plt.subplot((313))
        plt.plot(half_freqs, half_powers_mean)
        plt.vlines(half_freqs[signal_sfft_maxidx_mean], 0, np.max(half_powers_mean) * 1.2, colors='r', linestyles='--',
                   label='HR_mean ' + str(HR_mean))
        plt.scatter(half_freqs[signal_sfft_maxidx_mean], half_powers_mean[signal_sfft_maxidx_mean], alpha=0.5)
        plt.legend(loc='upper right')

        plt.show()
        plt.close()

    return HR_med, HR_mean

# 基于welch变换预测心率
# https://github.com/pavisj/rppg-pos/blob/master/pos_face_seg.py
def welch_est_hr(signal, fps, draw_fig_flag=False):
    signal = signal.flatten()
    half_freqs, half_power = welch(signal, fps, 'flattop', nperseg=len(signal),average='median')

    # 过滤干扰信号
    half_power[half_freqs < 0.7] = 0
    half_power[half_freqs > 3.0] = 0

    power_max_index = np.argmax(half_power)

    hr_freqs = half_freqs[power_max_index]
    hr_est = np.around(hr_freqs * 60, decimals=2)

    # draw_fig_flag = True
    if draw_fig_flag:
        print(f'welch_est_hr hr_est {hr_est}')
        win = int(len(half_freqs) / (12 * np.max(half_freqs)) + 1)
        # print('win', win)
        power_main_region = half_power[int(power_max_index - win):int(power_max_index + win + 1)]
        freq_main_reqion = half_freqs[int(power_max_index - win):int(power_max_index + win + 1)]

        # 比较原始和降噪后的信号与频谱
        plt.figure('welch_est_hr',figsize=(9,6))
        plt.subplot(211)
        plt.plot(signal, color='b', label='signal HR ' + str(hr_est))
        plt.legend(loc='upper right')

        plt.subplot(212)
        plt.plot(half_freqs[half_freqs < 5], half_power[half_freqs < 5], color='b')
        plt.vlines(hr_freqs, 0, int(np.max(half_power) * 1.2), colors='r', linestyles='dashed',
                                  label='hr_freqs ' + str(np.round(hr_freqs, 2)))
        plt.plot(freq_main_reqion, power_main_region, 'r')

        plt.scatter(half_freqs[power_max_index], half_power[power_max_index], alpha=0.5)
        plt.legend(loc='upper right')

        plt.show()
        plt.close()

    return hr_est

# 基于傅里叶的多种心率预测方法
def fft_est_hr_list(signal, fps, draw_fig_flag=False):
    # # 快速傅里叶变换，计算心率值
    HR_fft_default, HR_fft_snr = fft_est_hr(signal, fps, draw_fig_flag=draw_fig_flag)

    # # 短时傅里叶变换，计算心率值
    HR_stft_med, HR_stft_mean = stft_est_hr(signal, fps, draw_fig_flag=draw_fig_flag)

    # welch 计算心率值
    HR_welch = welch_est_hr(signal, fps, draw_fig_flag=draw_fig_flag)


    HR_fft_list = [HR_fft_default,HR_fft_snr,HR_stft_med,HR_stft_mean,HR_welch]
    HR_fft_vc = np.around(np.std(HR_fft_list)/np.mean(HR_fft_list), 2)


    # draw_fig_flag = True
    if draw_fig_flag:
        def get_main_region(power_max_index):
            power_main_region = half_power[int(power_max_index - win):int(power_max_index + win + 1)]
            freq_main_reqion = half_freqs[int(power_max_index - win):int(power_max_index + win + 1)]
            signal_power = np.sum(power_main_region)
            fft_snr = signal_power / (np.sum(half_power) - signal_power)
            # power_sub_region = half_power[int(power_max_index - win) * 2:int(power_max_index + win + 1) * 2]
            # freq_sub_reqion = half_freqs[int(power_max_index - win) * 2:int(power_max_index + win + 1) * 2]
            return power_main_region, freq_main_reqion, fft_snr

        # # 快速傅里叶变换
        half_freqs, half_power, win = get_fft_freq_power(signal, fps)


        # 默认最大频谱索引值
        power_max_default = np.argmax(half_power)

        power_sum_idx = get_power_maxidx(signal, fps, method='fft', draw_subfig_flag=False)

        # 比较原始和降噪后的信号与频谱
        plt.figure('fft_est_hr_list',figsize=(9,6))
        plt.subplot(311)
        plt.plot(signal, label='signal HR ' + str(HR_fft_list))
        plt.legend(loc='upper right')

        # power max default
        power_main_region, freq_main_reqion, fft_snr = get_main_region(power_max_default)
        hr_freqs = half_freqs[power_max_default]
        plt.subplot(312)
        plt.plot(half_freqs[half_freqs < 5], half_power[half_freqs < 5], label='SNR default ' + str(np.round(fft_snr, 2)))
        plt.vlines(hr_freqs, 0, int(np.max(half_power) * 1.2), colors='r', linestyles='dashed',
                                  label='hr_freqs ' + str(np.round(hr_freqs, 2)))
        plt.plot(freq_main_reqion, power_main_region, 'r')
        # plt.plot(freq_sub_reqion, power_sub_region, 'r')
        plt.scatter(half_freqs[power_max_default], half_power[power_max_default], alpha=0.5)
        plt.legend(loc='upper right')

        # power sum max
        power_main_region_,freq_main_reqion_,fft_snr_ = get_main_region(power_sum_idx)
        hr_freqs = half_freqs[power_sum_idx]
        plt.subplot(313)
        plt.plot(half_freqs[half_freqs < 5], half_power[half_freqs < 5], label='SNR max ' + str(np.round(fft_snr_, 2)))
        plt.vlines(hr_freqs, 0, int(np.max(half_power) * 1.2), colors='r', linestyles='dashed',
                                  label='hr_freqs ' + str(np.round(hr_freqs, 2)))
        plt.plot(freq_main_reqion_, power_main_region_, 'r')
        # plt.plot(freq_sub_reqion, power_sub_region, 'r')
        plt.scatter(half_freqs[power_sum_idx], half_power[power_sum_idx], alpha=0.5)
        plt.legend(loc='upper right')

        plt.show()
        plt.close()

    return HR_fft_list, HR_fft_vc

def PPG2HR(signal, fps, htr=0.8,  draw_fig_flag=False):
    # HR_fft_list = [HR_fft_default, HR_fft_snr, HR_stft_med, HR_stft_mean, HR_welch]

    HR_fft_list, HR_fft_vc = fft_est_hr_list(signal, fps, draw_fig_flag=draw_fig_flag)

    # IBT_feature = [AVPP, SDPP, PPVC]
    # HR_ibt_list = [HR_ibt_med, HR_ibt_mean]

    HR_ibt_list, IBT_feature, ret = ibt_est_hr(signal, fps, htr=htr, draw_fig_flag=draw_fig_flag)

    # HR_merge_list = [HR_fft_default, HR_fft_snr, HR_stft_med, HR_stft_mean, HR_welch,HR_ibt_med, HR_ibt_mean]
    HR_merge_list = HR_fft_list + HR_ibt_list
    HR_vc = np.std(HR_merge_list)/np.mean(HR_merge_list)
    HR_median = np.median(HR_merge_list)

    # HR_feature = [HR_vc, HR_fft_vc, ret]
    HR_feature = [HR_vc, HR_fft_vc, ret]

    # return HR_median,HR_feature
    return HR_median, HR_merge_list, IBT_feature, HR_feature

file_path ='./data/1_P026.h5'
hf = h5.File(file_path)
signal = hf['data_face_C3'] # C1:工业相机,C2:网络摄像头,CT:手机？

signal=signal[:,0,:]
fps = 30
win_time = 30
step_time = 7
interval_length = int(1.6 * fps)
order = 6

bvp_signal = get_bvp_signal(signal, fps, order, interval_length, chn=3, method='chrom')

HR_med, HR_mean = stft_est_hr(bvp_signal, fps, draw_fig_flag=True)
print(HR_med)