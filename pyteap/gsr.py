import numpy as np
import sys
sys.path.append('/kaggle/working/cogload/pyteap')
from filters import lowpass_mean_filter

def find_gsr_peaks(gsr, sr):
    mean_gsr = np.mean(gsr)
    std_gsr = np.std(gsr)
    threshold = mean_gsr + 2 * std_gsr
    t_low, t_up = 1, 10
    dn = np.diff((np.diff(gsr) <= 0).astype(int))
    idx_low = np.flatnonzero(dn < 0) + 1
    idx_high = np.flatnonzero(dn > 0) + 1

    rise_time, amp_peaks, pos_peaks = [], [], []
    for i in range(len(idx_low)):
        peaks = idx_high[idx_high < idx_low[i]]
        if peaks.size != 0:
            nearest = peaks[-1]
            # Kiểm tra xem nearest và idx_low[i] có nằm trong phạm vi hợp lệ không
            if nearest < len(gsr) and idx_low[i] < len(gsr):
                rt = (idx_low[i] - nearest) / sr  # rt là số thực, không phải Series
                amp = gsr.iloc[nearest] - gsr.iloc[idx_low[i]]  # Sử dụng iloc để lấy giá trị đơn lẻ từ Series
                # Kiểm tra điều kiện
                if (float(rt) >= t_low) and (float(rt) <= t_up) and (amp >= threshold).all:
                    rise_time.append(rt)
                    amp_peaks.append(amp)
                    pos_peaks.append(idx_low[i])

    return len(pos_peaks), amp_peaks, rise_time, pos_peaks


def get_gsr_features(gsr, sr):
    n_peaks, amp_peaks, rise_time, _ = find_gsr_peaks(gsr, sr)
    peaks_per_sec = (n_peaks * sr) / len(gsr)
    mean_amp = np.nanmax([np.mean(amp_peaks), 0])
    mean_risetime = np.nanmax([np.mean(rise_time), 0])
    mean_gsr = np.mean(gsr)
    std_gsr = np.std(gsr, ddof=1)

    return [peaks_per_sec, mean_amp, mean_risetime, mean_gsr, std_gsr]


def acquire_gsr(sig, sr, conversion=False):
    # check if conversion from siemens to ohm is needed
    if conversion:
        sig = conversion / sig
    elif (min(sig) >= 0) and (max(sig) < 1):
        sig = 1 / sig

    if round(sr / 16) > 0:
        sig = lowpass_mean_filter(sig, round(sr / 16))

    return sig
