# run_h3.py
"""
实验 H3：不同融合算法对比

对比方法：
- B4  MV-Fusion       三通道多数投票（硬判决后投票）
- B5  DS-Equal        标准D-S融合，三通道等权重 (w=1/3)
- B6  DS-Static        D-S融合+静态权重 (w_v=0.45, w_m=0.30, w_u=0.25)

实验设计：
- 复用 H2 的 7 个场景（单/双/全通道失效）
- 每个场景获取窗口级单通道预测概率，然后应用上述三种融合
- 输出场景汇总和各窗口级 CSV
"""
import csv
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from CV_channel_backend import ParkingDataset
from config import *
from detectors import GeomagDetector, UltrasonicDetector, VisualDetector
from sim_events import generate_event_sequence
from sim_geomag import GeomagSimulator
from sim_ultrasonic import UltrasonicSimulator
from sim_visual import degrade_lowlight


# ==================== 视觉图片池加载 ====================
def load_vision_pools(index_file, max_samples=2000):
    dataset = ParkingDataset(index_file, transform=None, max_samples=max_samples)
    occupied, empty = [], []
    for i in range(len(dataset)):
        img, label = dataset[i]
        if label == 1:
            occupied.append(img)
        else:
            empty.append(img)
    return occupied, empty


def compute_brightness(img_np):
    return np.mean(img_np) / 255.0


def _vision_sample_task(win_idx, true_label, occupied_pool, empty_pool,
                        v_gamma, v_noise_sigma, rng):
    pool = occupied_pool if true_label == 1 else empty_pool
    idx = rng.randint(len(pool))
    raw_img = pool[idx]
    img_np = np.array(raw_img)
    degraded = degrade_lowlight(img_np, v_gamma, v_noise_sigma)
    brightness = compute_brightness(degraded)
    return win_idx, degraded, brightness


def _batch_predict(det_v, images_np, transform, device, batch_size=128):
    all_probs = []
    for i in range(0, len(images_np), batch_size):
        batch_np = images_np[i:i + batch_size]
        batch_tensors = [transform(Image.fromarray(img)).unsqueeze(0) for img in batch_np]
        batch = torch.cat(batch_tensors, dim=0).to(device)
        with torch.no_grad():
            outputs = det_v.model(batch)
            probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs)


# ==================== 获取窗口级概率 ====================
def collect_window_probabilities(
        s_total, prob_m, y_m, prob_u, y_u,
        det_v, win_len_samples,
        v_gamma, v_noise_sigma,
        occupied_pool, empty_pool
):
    """返回各窗口的单通道概率、标签和单通道准确率"""
    n_windows = len(s_total) // win_len_samples
    n_10s = win_len_samples // 100

    # 地磁/超声聚合
    prob_m_resh = prob_m[:n_windows * n_10s].reshape(n_windows, n_10s)
    prob_u_resh = prob_u[:n_windows * n_10s].reshape(n_windows, n_10s)
    y_m_resh = y_m[:n_windows * n_10s].reshape(n_windows, n_10s)
    y_u_resh = y_u[:n_windows * n_10s].reshape(n_windows, n_10s)

    P_m = np.mean(prob_m_resh, axis=1)
    P_u = np.mean(prob_u_resh, axis=1)
    acc_m = np.mean((prob_m_resh > 0.5) == y_m_resh, axis=1)
    acc_u = np.mean((prob_u_resh > 0.5) == y_u_resh, axis=1)

    # 窗口标签
    window_labels = np.array([
        1 if np.mean(s_total[i * win_len_samples:(i + 1) * win_len_samples]) > 0.5 else 0
        for i in range(n_windows)
    ])

    # 视觉：多线程退化 + 批量推理
    rng = np.random.RandomState(42)
    futures = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for win_idx in range(n_windows):
            lbl = window_labels[win_idx]
            future = executor.submit(_vision_sample_task,
                                     win_idx, lbl, occupied_pool, empty_pool,
                                     v_gamma, v_noise_sigma, rng)
            futures.append(future)
        win_indices = np.empty(n_windows, dtype=int)
        degraded_images = [None] * n_windows
        for future in tqdm(as_completed(futures), total=n_windows, desc='视觉退化'):
            wid, img_np, _ = future.result()
            win_indices[wid] = wid
            degraded_images[wid] = img_np
    sorted_idx = np.argsort(win_indices)
    degraded_images = [degraded_images[i] for i in sorted_idx]

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    P_v = _batch_predict(det_v, degraded_images, eval_transform, DEVICE, batch_size=128)
    acc_v = ((P_v > 0.5).astype(int) == window_labels).astype(float)

    return P_v, P_m, P_u, window_labels, acc_v, acc_m, acc_u


# ==================== 融合方法实现 ====================
def ds_combine(m1, m2):
    """两个 BPA 的 Dempster 组合（向量化）"""
    H1_1, H2_1, Theta_1 = m1[:, 0], m1[:, 1], m1[:, 2]
    H1_2, H2_2, Theta_2 = m2[:, 0], m2[:, 1], m2[:, 2]
    K = H1_1 * H2_2 + H2_1 * H1_2
    safe = K < 1.0
    H1 = np.where(safe, (H1_1 * H1_2 + H1_1 * Theta_2 + Theta_1 * H1_2) / (1 - K), 0.0)
    H2 = np.where(safe, (H2_1 * H2_2 + H2_1 * Theta_2 + Theta_1 * H2_2) / (1 - K), 0.0)
    Theta = np.where(safe, Theta_1 * Theta_2 / (1 - K), 1.0)
    return np.stack([H1, H2, Theta], axis=1)


def ds_fusion_fixed_weights(P_v, P_m, P_u, w_v, w_m, w_u):
    """固定权重 D‑S 融合（标准组合，无冲突检测）"""
    n = len(P_v)
    m_v = np.column_stack([P_v * w_v, (1 - P_v) * w_v, np.full(n, 1 - w_v)])
    m_m = np.column_stack([P_m * w_m, (1 - P_m) * w_m, np.full(n, 1 - w_m)])
    m_u = np.column_stack([P_u * w_u, (1 - P_u) * w_u, np.full(n, 1 - w_u)])
    m12 = ds_combine(m_v, m_m)
    m_final = ds_combine(m12, m_u)
    H1, H2 = m_final[:, 0], m_final[:, 1]
    pred = np.full(n, -1, dtype=int)
    pred[H1 > 0.6] = 1
    pred[H2 > 0.6] = 0
    return pred


def majority_vote(P_v, P_m, P_u):
    """硬判决后多数投票"""
    pred_v = (P_v > 0.5).astype(int)
    pred_m = (P_m > 0.5).astype(int)
    pred_u = (P_u > 0.5).astype(int)
    # 计算总和，>=2 则为占用
    votes = pred_v + pred_m + pred_u
    return (votes >= 2).astype(int)


# ==================== 退化参数与场景 ====================
def get_mild_params():
    return {'v_gamma': 0.6, 'v_noise_sigma': 20,
            'emi': 20, 'drift': 10,
            'multipath_p': 0.5, 'condensation_sev': 4}


# ==================== 退化条件下运行 ====================
def generate_signals_by_scene(scene, s_total):
    p = get_mild_params()
    normal = {'v_gamma': None, 'v_noise_sigma': None, 'emi': 0, 'drift': 0, 'multipath_p': 0.0, 'condensation_sev': 0}
    cfg = normal.copy()
    if 'v' in scene:
        cfg['v_gamma'] = p['v_gamma']
        cfg['v_noise_sigma'] = p['v_noise_sigma']
    if 'm' in scene:
        cfg['emi'] = p['emi']
        cfg['drift'] = p['drift']
    if 'u' in scene:
        cfg['multipath_p'] = p['multipath_p']
        cfg['condensation_sev'] = p['condensation_sev']

    seed_offset = hash(scene) % 10
    # 地磁
    mag_sim = GeomagSimulator(seed=114 + seed_offset)
    B_clean = mag_sim.generate_baseline(168 * 3600)
    B_clean = mag_sim.inject_self_occupancy(B_clean, s_total)
    if cfg['emi'] > 0:
        B_signal = mag_sim.inject_em_interference(B_clean, A_total=cfg['emi'])
        B_signal = mag_sim.inject_drift(B_signal, severe=cfg['drift'])
    else:
        B_signal = B_clean

    # 超声
    us_sim = UltrasonicSimulator(seed=1145 + seed_offset)
    d_clean = us_sim.generate_base_distance(s_total)
    if cfg['multipath_p'] > 0:
        d_signal = us_sim.inject_multipath(d_clean, p=cfg['multipath_p'])
        d_signal = us_sim.inject_white_noise(d_signal)
        d_signal = us_sim.inject_condensation(d_signal, severity=cfg['condensation_sev'])
    else:
        d_signal = us_sim.inject_white_noise(d_clean)

    return B_signal, d_signal, cfg


# ==================== 主函数 ====================
def main():
    print("=" * 20)
    print("H3 实验：融合算法对比")
    print("=" * 20)

    s_total = generate_event_sequence(168, weekday=True, seed=42)
    print(f"占用序列长度: {len(s_total):,}")

    # 加载检测器
    det_v = VisualDetector(CV_MODEL_PATH)
    with open('models/geomag_svm_model.pkl', 'rb') as f:
        det_m = GeomagDetector()
        det_m.svm = pickle.load(f)
    with open('models/us_params.json', 'r') as f:
        us_params = json.load(f)
    det_u = UltrasonicDetector(d_threshold=us_params['d_threshold'],
                               alpha=us_params['alpha'])

    print("加载视觉图片池...")
    occupied_pool, empty_pool = load_vision_pools(PUC_INDEX_FILE, max_samples=2000)

    scenes = ['v', 'm', 'u', 'vm', 'vu', 'mu', 'vmu']
    # scenes = ['m', 'mv', 'mu']
    win_len_samples = WINDOW_SEC * FS  # 6000

    all_summaries = []

    for scene in scenes:
        print(f"\n{'=' * 20} 场景: {scene} {'=' * 20}")
        B_signal, d_signal, cfg = generate_signals_by_scene(scene, s_total)

        # 提取环境参数（仅用于记录，融合时不使用）
        v_gamma = cfg['v_gamma'] if cfg['v_gamma'] is not None else 1.0
        v_noise = cfg['v_noise_sigma'] if cfg['v_noise_sigma'] is not None else 0.0

        # 计算10秒窗口概率
        prob_m, y_m = det_m.process_segment(B_signal[:len(B_signal) // 100 * 100],
                                            s_total[:len(B_signal) // 100 * 100])
        n_u = len(d_signal) // 100
        prob_u = np.zeros(n_u)
        y_u = np.zeros(n_u)
        for i in range(n_u):
            seg = d_signal[i * 100:(i + 1) * 100]
            d_med = np.nanmedian(seg) if not np.isnan(seg).all() else H_CEILING
            prob_u[i] = det_u.predict_proba(d_med)
            y_u[i] = 1 if np.mean(s_total[i * 100:(i + 1) * 100]) > 0.5 else 0

        # 获取窗口概率与单通道准确率
        P_v, P_m, P_u, labels, acc_v, acc_m, acc_u = collect_window_probabilities(
            s_total, prob_m, y_m, prob_u, y_u,
            det_v, win_len_samples,
            v_gamma=v_gamma, v_noise_sigma=v_noise,
            occupied_pool=occupied_pool, empty_pool=empty_pool
        )

        # 1. 多数投票
        pred_mv = majority_vote(P_v, P_m, P_u)
        acc_mv = (pred_mv == labels).astype(float)

        # 2. 等权重 D-S
        pred_ds_eq = ds_fusion_fixed_weights(P_v, P_m, P_u, 1 / 3, 1 / 3, 1 / 3)
        acc_ds_eq = (pred_ds_eq == labels).astype(float)

        # 3. 静态权重 D-S (0.45, 0.30, 0.25)
        pred_ds_static = ds_fusion_fixed_weights(P_v, P_m, P_u, 0.45, 0.30, 0.25)
        acc_ds_static = (pred_ds_static == labels).astype(float)

        # 写入窗口 CSV
        window_csv = os.path.join(RESULTS_DIR, f'h3_scene_{scene}_window.csv')
        with open(window_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['win_idx', 'label',
                             'acc_v', 'acc_m', 'acc_u',
                             'acc_mv', 'acc_ds_eq', 'acc_ds_static',
                             'P_v', 'P_m', 'P_u'])
            for i in range(len(labels)):
                writer.writerow([i, labels[i],
                                 acc_v[i], acc_m[i], acc_u[i],
                                 acc_mv[i], acc_ds_eq[i], acc_ds_static[i],
                                 P_v[i], P_m[i], P_u[i]])

        # 汇总
        best_single = max(np.mean(acc_v), np.mean(acc_m), np.mean(acc_u))
        summary = {
            'scene': scene,
            'acc_v_mean': np.mean(acc_v),
            'acc_m_mean': np.mean(acc_m),
            'acc_u_mean': np.mean(acc_u),
            'acc_mv_mean': np.mean(acc_mv),
            'acc_ds_eq_mean': np.mean(acc_ds_eq),
            'acc_ds_static_mean': np.mean(acc_ds_static),
            'best_single': best_single,
            'gain_mv_pp': (np.mean(acc_mv) - best_single) * 100,
            'gain_ds_eq_pp': (np.mean(acc_ds_eq) - best_single) * 100,
            'gain_ds_static_pp': (np.mean(acc_ds_static) - best_single) * 100
        }
        all_summaries.append(summary)
        print(f"  单通道: V={summary['acc_v_mean']:.3f} M={summary['acc_m_mean']:.3f} U={summary['acc_u_mean']:.3f}")
        print(
            f"  MV: {summary['acc_mv_mean']:.3f}  DS-Eq: {summary['acc_ds_eq_mean']:.3f}  DS-Static: {summary['acc_ds_static_mean']:.3f}")

    # 保存汇总
    summary_df = pd.DataFrame(all_summaries)
    summary_path = os.path.join(RESULTS_DIR, 'h3_results.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nH3 汇总结果已保存至 {summary_path}")


if __name__ == "__main__":
    main()
