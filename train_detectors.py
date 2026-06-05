# train_detectors.py
"""
阶段② 检测器训练
- 地磁 SVM：使用一周正常工况仿真数据，采用干净 EMA 背景特征，保存模型。
- 超声 Sigmoid：网格搜索最佳 d_threshold 与 alpha，保存参数。
"""
import json
import pickle

from sklearn.metrics import accuracy_score

from config import *
from detectors import GeomagDetector, UltrasonicDetector
from feature_engineering import compute_clean_background, extract_geomag_features_clean
from sim_events import generate_event_sequence
from sim_geomag import GeomagSimulator
from sim_ultrasonic import UltrasonicSimulator


# --------------- 超声特征提取 ---------------
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

    # ========== 地磁 SVM 训练 ==========
    print("【地磁 SVM 训练】生成一周正常仿真数据...")
    s_train = generate_event_sequence(24 * 7, fs=FS, weekday=True, seed=100)
    mag_sim = GeomagSimulator(seed=100)
    B_train = mag_sim.generate_baseline(len(s_train) / FS)
    B_train = mag_sim.inject_self_occupancy(B_train, s_train)  # 加强版注入函数

    # 干净背景 + 特征提取
    B_bg = compute_clean_background(B_train, s_train)
    X_train, y_train = extract_geomag_features_clean(B_train, s_train, B_bg)
    print(f"原始训练样本数：{len(X_train)}")

    # 均衡下采样至约 5000 条
    if len(X_train) > 5000:
        idx1 = np.where(y_train == 1)[0]
        idx0 = np.where(y_train == 0)[0]
        n = min(2500, len(idx1), len(idx0))
        np.random.seed(42)
        idx = np.concatenate([
            np.random.choice(idx1, n, replace=False),
            np.random.choice(idx0, n, replace=False)
        ])
        X_train, y_train = X_train[idx], y_train[idx]
    print(f"均衡后训练样本数：{len(X_train)}")

    # 训练 SVM
    det_mag = GeomagDetector(ref_B=None)  # ref_B 由特征提取函数动态提供
    det_mag.fit(X_train, y_train)
    # 检查训练集准确率
    train_accuracy = det_mag.svm.score(X_train, y_train)
    print(f"模型在训练集上的准确率: {train_accuracy:.4f}")
    # 保存模型
    with open('models/geomag_svm_model.pkl', 'wb') as f:
        pickle.dump(det_mag.svm, f)
    print("地磁 SVM 模型已保存为 geomag_svm_model.pkl")

    # ========== 超声 Sigmoid 调参 ==========
    print("\n【超声 Sigmoid 调参】生成一天验证数据...")
    s_val = generate_event_sequence(24, fs=FS, weekday=True, seed=200)
    us_sim = UltrasonicSimulator(seed=200)
    d_val = us_sim.generate_base_distance(s_val)
    X_val, y_val = extract_us_features(d_val, s_val)

    # 网格搜索
    best_acc = 0
    best_params = {'d_threshold': 1.85, 'alpha': 8.0}
    for d_th in np.arange(1.6, 2.05, 0.05):
        for alpha in np.arange(5.0, 15.5, 0.5):
            det_tmp = UltrasonicDetector(d_threshold=d_th, alpha=alpha)
            prob = det_tmp.predict_proba(X_val.flatten())
            pred = (prob > 0.5).astype(int)
            acc = accuracy_score(y_val, pred)
            if acc > best_acc:
                best_acc = acc
                best_params = {'d_threshold': d_th, 'alpha': alpha}
    print(f"最佳参数：d_threshold={best_params['d_threshold']:.2f}, "
          f"alpha={best_params['alpha']:.1f}, 验证准确率={best_acc:.4f}")

    with open('models/us_params.json', 'w') as f:
        json.dump(best_params, f)
    print("超声参数已保存为 us_params.json")


if __name__ == "__main__":
    main()
