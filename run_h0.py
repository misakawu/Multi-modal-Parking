# run_h0.py
"""
阶段② 正常工况验收
加载地磁 SVM 模型和超声参数，在独立一天测试数据上评估。
地磁检测器内部使用因果 EMA 背景（仅依赖历史数据），对外暴露 predict_occupancy。
输出 Accuracy 和 F1 到 results/h0_normal_accuracy.csv，并打印到控制台。
"""
import json
import pickle

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from config import *
from detectors import GeomagDetector, UltrasonicDetector
from sim_events import generate_event_sequence
from sim_geomag import GeomagSimulator
from sim_ultrasonic import UltrasonicSimulator


# ---------- 超声特征提取 ----------
def extract_us_features(d, s, win_len=10):
    """每个窗口取中位数距离作为特征"""
    N = len(s)
    X, y = [], []
    for i in range(0, N - win_len + 1, win_len):
        win_d = d[i:i + win_len]
        win_s = s[i:i + win_len]
        d_med = np.nanmedian(win_d)
        if np.isnan(d_med):
            d_med = H_CEILING
        X.append([d_med])
        label = 1 if np.mean(win_s) > 0.5 else 0
        y.append(label)
    return np.array(X), np.array(y)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---------- 加载模型 / 参数 ----------
    with open('models/geomag_svm_model.pkl', 'rb') as f:
        svm_model = pickle.load(f)
    # 初始化地磁检测器（内部会采用流式背景更新）
    det_mag = GeomagDetector()
    det_mag.svm = svm_model

    with open('models/us_params.json', 'r') as f:
        us_params = json.load(f)
    det_us = UltrasonicDetector(d_threshold=us_params['d_threshold'],
                                alpha=us_params['alpha'])

    # ---------- 生成独立测试数据 (1 天) ----------
    print("生成一天测试数据...")
    seed_test = 300
    s_test = generate_event_sequence(24, fs=FS, weekday=True, seed=seed_test)

    # ---------- 地磁评估 ----------
    mag_sim = GeomagSimulator(seed=seed_test)
    B_test = mag_sim.generate_baseline(len(s_test) / FS)
    B_test = mag_sim.inject_self_occupancy(B_test, s_test)

    # 一键式预测（内部自动完成因果背景更新和特征提取）
    prob_mag, y_mag = det_mag.predict_occupancy(B_test, s_test)
    pred_mag = (prob_mag > 0.5).astype(int)
    mag_acc = accuracy_score(y_mag, pred_mag)
    mag_f1 = f1_score(y_mag, pred_mag, zero_division=0)

    # ---------- 超声评估 ----------
    us_sim = UltrasonicSimulator(seed=seed_test)
    d_test = us_sim.generate_base_distance(s_test)
    X_us, y_us = extract_us_features(d_test, s_test)
    prob_us = det_us.predict_proba(X_us.flatten())
    pred_us = (prob_us > 0.5).astype(int)
    us_acc = accuracy_score(y_us, pred_us)
    us_f1 = f1_score(y_us, pred_us, zero_division=0)

    # ---------- 控制台输出 ----------
    print("\n========== 正常工况验收结果 ==========")
    print(f"地磁通道 (Geomag)  : Accuracy = {mag_acc:.4f}, F1 = {mag_f1:.4f}")
    print(f"超声通道 (Ultrason) : Accuracy = {us_acc:.4f}, F1 = {us_f1:.4f}")
    print("======================================")

    # ---------- 保存 CSV ----------
    df = pd.DataFrame({
        'channel': ['geomag', 'ultrasonic'],
        'accuracy': [mag_acc, us_acc],
        'f1': [mag_f1, us_f1],
    })
    csv_path = os.path.join(RESULTS_DIR, 'h0_normal_accuracy.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n结果已保存至 {csv_path}")


if __name__ == "__main__":
    main()
