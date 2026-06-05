# feature_engineering.py
import numpy as np

from config import FS


def compute_clean_background(B_full, s, tau=600, fs=FS, B_init=None):
    """
    基于空闲时段指数移动平均（EMA）的背景估计（因果，无未来信息）。
    B_full: (N,3) 磁场时序
    s: (N,) 占用真值（0/1）
    tau: 时间常数（秒）
    fs: 采样率
    B_init: 初始背景 (3,)，若为None则用第一个采样点
    返回: B_bg (N,3) 与 B_full 等长的背景序列
    """
    N = len(B_full)
    alpha = 1 - np.exp(-1.0 / (tau * fs))
    B_bg = np.zeros((N, 3))
    if B_init is None:
        B_bg[0] = B_full[0]
    else:
        # 初始时刻的背景由外部给定，但第一个采样点仍需要决定是否更新
        # 约定：在时刻0之前的外部背景作为第-1时刻的背景
        # 这里我们用一个临时变量表示上一时刻的背景
        prev_bg = B_init
        # 对于时刻0，如果s[0] == 0，则用B_full[0]更新，否则保持prev_bg
        if s[0] == 0:
            B_bg[0] = alpha * B_full[0] + (1 - alpha) * prev_bg
        else:
            B_bg[0] = prev_bg
        # 继续处理后续时刻
        for t in range(1, N):
            if s[t - 1] == 0:  # 前一个时刻空闲，更新背景
                B_bg[t] = alpha * B_full[t] + (1 - alpha) * B_bg[t - 1]
            else:
                B_bg[t] = B_bg[t - 1]
        return B_bg

    for t in range(1, N):
        if s[t - 1] == 0:
            B_bg[t] = alpha * B_full[t] + (1 - alpha) * B_bg[t - 1]
        else:
            B_bg[t] = B_bg[t - 1]
    return B_bg


def extract_geomag_features_clean(B_full, s, B_bg, win_len=100):
    """
    基于干净背景提取滑动窗口特征。
    B_full: (N,3)
    s: (N,)
    B_bg: (N,3) 预先计算的背景序列
    win_len: 窗口点数
    返回 X (n_windows,3), y (n_windows,)
    """
    N = len(s)
    X, y = [], []
    for start in range(0, N - win_len + 1, win_len):
        end = start + win_len
        ref_B = B_bg[start]  # 窗口起始时刻的干净背景
        win_B = B_full[start:end]
        delta = np.linalg.norm(win_B - ref_B, axis=1)
        feat = [np.mean(delta), np.var(delta), np.max(delta) - np.min(delta)]
        X.append(feat)
        label = 1 if np.mean(s[start:end]) > 0.5 else 0
        y.append(label)
    return np.array(X), np.array(y)
