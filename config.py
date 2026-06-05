# config.py
import os

import numpy as np
import torch

# ========== 通用 ==========
SEED = 42
np.random.seed(SEED)
FS = 10  # 采样率 (Hz)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPU_CORE = 12

# ========== 路径（请按实际修改） ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = BASE_DIR
CV_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models/cv_best_model.pth')
PKLOT_INDEX_DIR = r'E:\DATASET\PKLot\index'  # 请修改为实际路径
PUC_INDEX_FILE = os.path.join(PKLOT_INDEX_DIR, 'puc_test.json')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ========== 地磁 ==========
B_EARTH = np.array([28.5, -1.5, 38.0])  # μT，成都 (北,东,下)
SIGMA_N = 0.3
# SIGMA_DRIFT = 0.03
SIGMA_DRIFT = 0.0017
M0_MEAN, M0_STD = 2.0, 1.0
M0_MIN, M0_MAX = 0.3, 6.0
NEIGHBOR_DIST = [2.5, 5.0, 5.5]  # m
EM_AMPLITUDE_RANGE = [5, 30]
EM_HARMONIC_FREQ = [1, 2, 5, 10]  # kHz
EM_HARMONIC_WEIGHTS = [1.0, 0.5, 0.3, 0.2]

# ========== 超声 ==========
H_CEILING = 2.4
SIGMA_D_ULTRASONIC = 0.02
VEHICLE_TYPE_DIST = [
    {'name': 'sedan', 'p': 0.60, 'h_mean': 1.45, 'h_std': 0.05, 'h_min': None, 'h_max': None},
    {'name': 'suv', 'p': 0.25, 'h_mean': 1.65, 'h_std': 0.07, 'h_min': None, 'h_max': None},
    {'name': 'pickup', 'p': 0.08, 'h_mean': 1.85, 'h_std': 0.10, 'h_min': None, 'h_max': 1.95},
    {'name': 'sportscar', 'p': 0.05, 'h_mean': 1.30, 'h_std': 0.04, 'h_min': None, 'h_max': None},
    {'name': 'abnormal', 'p': 0.02, 'h_mean': None, 'h_std': None, 'h_min': 1.95, 'h_max': 2.15}
]

# ========== 视觉退化范围 ==========
GAMMA_RANGE = [0.2, 0.7]
NOISE_SIGMA_RANGE = [5, 30]
OIL_DIRT_RATIO_RANGE = [0.05, 0.40]
WATER_MIST_BETA_RANGE = [0.5, 2.5]
GLARE_RADIUS_RANGE = [30, 80]
GLARE_INTENSITY_RANGE = [60, 120]

# ========== 滚动窗口与阈值 ==========
WINDOW_SEC = 10 * 60  # 10 分钟
ACC_THRESHOLD = 0.85
