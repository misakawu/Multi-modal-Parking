# detectors.py
import torch.nn as nn
from sklearn.svm import SVC
from torchvision import models

from config import *


# ---------- 视觉检测器 ----------
# detectors.py（修改片段）

class VisualDetector:
    def __init__(self, model_path=CV_MODEL_PATH, num_classes=2, device=DEVICE):
        self.model = models.mobilenet_v3_small(weights=None)
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(device)
        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=False)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict_proba(self, img):
        """ 输入 PIL Image，返回占用概率 """
        return self.model(img)


# ---------- 超声检测器 ----------
class UltrasonicDetector:
    def __init__(self, d_threshold=1.85, alpha=15.0):
        self.d_threshold = d_threshold
        self.alpha = alpha

    def predict_proba(self, d):
        """ 输入测距值 (标量或数组)，返回占用概率 """
        d = np.asarray(d, dtype=float)
        d[np.isnan(d)] = H_CEILING  # 缺失值用空位距离代替
        return 1.0 / (1.0 + np.exp(-self.alpha * (self.d_threshold - d)))


# ---------- 地磁检测器 ----------
from feature_engineering import compute_clean_background, extract_geomag_features_clean


class GeomagDetector:
    def __init__(self, window_len=100, ref_B=None, tau=600):
        self.window_len = window_len
        self.ref_B = ref_B if ref_B is not None else B_EARTH
        self.svm = SVC(kernel='rbf', C=1.0, probability=True)
        self.tau = tau  # 背景 EMA 时间常数
        self._B_bg = None  # 内部背景状态 (3,)

    # ---------- 原有的方法（仍可用） ----------
    def compute_features(self, B_window):
        delta = np.linalg.norm(B_window - self.ref_B, axis=1)
        return np.array([np.mean(delta), np.var(delta), np.max(delta) - np.min(delta)])

    def fit(self, X, y):
        self.svm.fit(X, y)

    def predict_proba(self, X):
        return self.svm.predict_proba(X)[:, 1]

    # ---------- 流式处理接口 ----------
    def reset_background(self, B_init=None):
        """重置内部背景，若未提供则下次将从第一帧自动初始化"""
        self._B_bg = B_init

    def process_segment(self, B_segment, s_segment):
        """
        处理一段连续地磁数据，返回该段内所有窗口的占用概率及标签。
        背景更新完全基于历史空闲数据，不使用未来信息。
        B_segment: (L,3) 本段磁场采样
        s_segment: (L,) 本段占用真值（0/1）
        返回:
            prob: (n_windows,) 占用概率
            y_true: (n_windows,) 多数投票标签
        """
        # 1. 计算本段背景（衔接之前的状态）
        B_bg = compute_clean_background(B_segment, s_segment, tau=self.tau, fs=10, B_init=self._B_bg)
        # 2. 提取特征
        X, y_true = extract_geomag_features_clean(B_segment, s_segment, B_bg, win_len=self.window_len)
        # 3. 预测
        prob = self.predict_proba(X)
        # 4. 更新内部背景为本段最后时刻的背景（留给下一段衔接）
        self._B_bg = B_bg[-1]
        return prob, y_true

    def predict_occupancy(self, B_full, s_true):
        """
        一次性处理全长数据（会重置背景，方便独立测试）。
        B_full: (N,3) 全部磁场序列
        s_true: (N,) 占用真值
        返回: prob, y_true
        """
        self.reset_background(None)
        return self.process_segment(B_full, s_true)
