# run_h4.py
"""
H4：多退化等级 × 多融合算法对比

实验设计：
- 7 个故障场景（v, m, u, vm, vu, mu, vmu）
- 5 个退化等级（0‑无退化, 1‑轻度, 2‑中度, 3‑重度, 4‑极重度）
- 每个组合评估：
   - 单通道准确率（视觉、地磁、超声）
   - 融合方法：MV（多数投票）、DS‑Equal（等权D-S）、DS‑Static（静态权重D-S）、DS‑Ours（自适应加权D-S）
- 输出每个组合的汇总指标到 h4_results.csv
- 各组合窗口级结果保存到 h4_scene_{scene}_lev{level}_window.csv

核心任务：
1. 自适应通道权重：基于传感器质量动态计算权重
   - 视觉：根据亮度使用sigmoid函数映射权重
   - 地磁：根据电磁干扰强度调整权重
   - 超声：根据多径效应严重程度调整权重

2. 冲突感知融合策略：
   - 低冲突：标准D-S证据理论组合
   - 高冲突：基于相对权重的选择性融合，放大最可靠通道的影响

3. 动态决策阈值：
   - 考虑不确定性、冲突度、置信度差距
   - 对未决策样本使用多数投票兜底
"""
import csv
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

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
                        v_gamma, rng):
    pool = occupied_pool if true_label == 1 else empty_pool
    idx = rng.randint(len(pool))
    raw_img = pool[idx]
    img_np = np.array(raw_img)
    degraded = degrade_lowlight(img_np, v_gamma)
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


# ==================== 通道权重计算====================
def compute_channel_weights(brightness_arr, multipath_p, emi):
    k_v, b_v = 15.0, 0.2
    k_u, b_u = 10.0, 0.45
    k_m, b_m = 0.5, 10.0
    n = len(brightness_arr)
    w_v = 1.0 / (1.0 + np.exp(-k_v * (brightness_arr - b_v)))
    if np.isscalar(multipath_p):
        w_u = np.full(n, 1.0 / (1.0 + np.exp(k_u * (multipath_p - b_u))))
    else:
        w_u = 1.0 / (1.0 + np.exp(k_u * (np.asarray(multipath_p) - b_u)))
    if np.isscalar(emi):
        w_m = np.full(n, 1.0 / (1.0 + np.exp(k_m * (emi - b_m))))
    else:
        w_m = 1.0 / (1.0 + np.exp(k_m * (np.asarray(emi) - b_m)))

    # 关键改进：确保最小权重，避免完全失效
    min_weight = 0.1
    w_v = np.maximum(w_v, min_weight)
    w_m = np.maximum(w_m, min_weight)
    w_u = np.maximum(w_u, min_weight)

    return w_v, w_u, w_m


# ==================== D‑S 融合 ====================
def ds_combine(m1, m2):
    H1_1, H2_1, Theta_1 = m1[:, 0], m1[:, 1], m1[:, 2]
    H1_2, H2_2, Theta_2 = m2[:, 0], m2[:, 1], m2[:, 2]
    K = H1_1 * H2_2 + H2_1 * H1_2
    safe = K < 1.0
    H1 = np.where(safe, (H1_1 * H1_2 + H1_1 * Theta_2 + Theta_1 * H1_2) / (1 - K), 0.0)
    H2 = np.where(safe, (H2_1 * H2_2 + H2_1 * Theta_2 + Theta_1 * H2_2) / (1 - K), 0.0)
    Theta = np.where(safe, Theta_1 * Theta_2 / (1 - K), 1.0)
    return np.stack([H1, H2, Theta], axis=1)


def ds_fusion_fixed_weights(P_v, P_m, P_u, w_v, w_m, w_u):
    n = len(P_v)
    if np.isscalar(w_v):
        w_v = np.full(n, w_v)
    if np.isscalar(w_m):
        w_m = np.full(n, w_m)
    if np.isscalar(w_u):
        w_u = np.full(n, w_u)

    m_v = np.column_stack([P_v * w_v, (1 - P_v) * w_v, 1 - w_v])
    m_m = np.column_stack([P_m * w_m, (1 - P_m) * w_m, 1 - w_m])
    m_u = np.column_stack([P_u * w_u, (1 - P_u) * w_u, 1 - w_u])
    m12 = ds_combine(m_v, m_m)
    m_final = ds_combine(m12, m_u)
    H1, H2 = m_final[:, 0], m_final[:, 1]

    # 动态阈值：根据不确定性调整
    Theta = m_final[:, 2]
    decision_threshold = 0.55 + 0.15 * Theta  # 不确定性越高，阈值越高
    pred = np.full(n, -1, dtype=int)
    pred[H1 > decision_threshold] = 1
    pred[H2 > decision_threshold] = 0
    return pred


def ds_fusion_decision_batch(P_v, P_m, P_u, w_v, w_u, w_m):
    n = len(P_v)
    m_v = np.column_stack([P_v * w_v, (1 - P_v) * w_v, 1 - w_v])
    m_m = np.column_stack([P_m * w_m, (1 - P_m) * w_m, 1 - w_m])
    m_u = np.column_stack([P_u * w_u, (1 - P_u) * w_u, 1 - w_u])

    # 计算两两冲突
    K_vm = m_v[:, 0] * m_m[:, 1] + m_v[:, 1] * m_m[:, 0]
    K_vu = m_v[:, 0] * m_u[:, 1] + m_v[:, 1] * m_u[:, 0]
    K_mu = m_m[:, 0] * m_u[:, 1] + m_m[:, 1] * m_u[:, 0]
    K_avg = (K_vm + K_vu + K_mu) / 3.0

    # 低冲突路径：标准D-S组合
    m12 = ds_combine(m_v, m_m)
    m_final_low = ds_combine(m12, m_u)

    # 高冲突路径改进：基于相对权重的选择性融合
    w_total = w_v + w_m + w_u
    mask_zero = w_total == 0
    w_total_safe = np.where(mask_zero, 1.0, w_total)

    # 归一化权重（相对可信度）
    w_v_norm = w_v / w_total_safe
    w_m_norm = w_m / w_total_safe
    w_u_norm = w_u / w_total_safe

    # 找出最可靠的通道（权重最高者）
    max_weights = np.maximum.reduce([w_v_norm, w_m_norm, w_u_norm])

    # 高冲突策略：偏向最可靠通道的加权组合
    # 放大最高权重的影响
    w_v_enhanced = np.where(w_v_norm == max_weights, w_v_norm * 1.5, w_v_norm)
    w_m_enhanced = np.where(w_m_norm == max_weights, w_m_norm * 1.5, w_m_norm)
    w_u_enhanced = np.where(w_u_norm == max_weights, w_u_norm * 1.5, w_u_norm)

    # 重新归一化
    w_sum_enhanced = w_v_enhanced + w_m_enhanced + w_u_enhanced
    w_v_enhanced /= w_sum_enhanced
    w_m_enhanced /= w_sum_enhanced
    w_u_enhanced /= w_sum_enhanced

    m_avg = np.column_stack([
        P_v * w_v_enhanced + P_m * w_m_enhanced + P_u * w_u_enhanced,
        (1 - P_v) * w_v_enhanced + (1 - P_m) * w_m_enhanced + (1 - P_u) * w_u_enhanced,
        1 - max_weights  # 不确定性由最可靠通道决定
    ])
    m_temp = ds_combine(m_avg, m_avg)
    m_final_high = ds_combine(m_temp, m_avg)

    # 自适应冲突阈值调整
    adaptive_conflict_threshold = 0.5 + 0.3 * np.exp(-0.5 * w_total)
    low_conflict = K_avg < adaptive_conflict_threshold

    H1 = np.where(low_conflict, m_final_low[:, 0], m_final_high[:, 0])
    H2 = np.where(low_conflict, m_final_low[:, 1], m_final_high[:, 1])

    # 改进的动态决策阈值
    Theta_low = m_final_low[:, 2]
    Theta_high = m_final_high[:, 2]
    Theta = np.where(low_conflict, Theta_low, Theta_high)

    # 新的阈值策略：更激进地做出决策
    # 基础阈值降低到0.45，允许更多决策
    uncertainty_penalty = 0.1 * Theta  # 减小不确定性的惩罚系数
    conflict_bonus = 0.15 * (1 - np.exp(-2 * K_avg))

    # 当有明确倾向时（H1或H2明显大于另一方），降低阈值
    confidence_gap = np.abs(H1 - H2)
    confidence_bonus = 0.1 * np.exp(-2 * confidence_gap)  # 差距越大，bonus越小

    decision_threshold = 0.45 + uncertainty_penalty - conflict_bonus + confidence_bonus
    decision_threshold = np.clip(decision_threshold, 0.40, 0.65)  # 收紧阈值范围

    pred = np.full(n, -1, dtype=int)
    pred[H1 > decision_threshold] = 1
    pred[H2 > decision_threshold] = 0

    # 后处理：对于未决策的样本，使用多数投票兜底
    undecided = pred == -1
    if np.any(undecided):
        pred_v = (P_v[undecided] > 0.5).astype(int)
        pred_m = (P_m[undecided] > 0.5).astype(int)
        pred_u = (P_u[undecided] > 0.5).astype(int)
        votes = pred_v + pred_m + pred_u
        pred[undecided] = (votes >= 2).astype(int)

    return pred


def majority_vote(P_v, P_m, P_u):
    pred_v = (P_v > 0.5).astype(int)
    pred_m = (P_m > 0.5).astype(int)
    pred_u = (P_u > 0.5).astype(int)
    votes = pred_v + pred_m + pred_u
    return (votes >= 2).astype(int)


# ==================== 窗口概率与亮度获取 ====================
def collect_window_data(
        s_total, prob_m, y_m, prob_u, y_u,
        det_v, win_len_samples,
        v_gamma,
        occupied_pool, empty_pool,
        return_brightness=False
):
    n_windows = len(s_total) // win_len_samples
    n_10s = win_len_samples // 100

    prob_m_resh = prob_m[:n_windows * n_10s].reshape(n_windows, n_10s)
    prob_u_resh = prob_u[:n_windows * n_10s].reshape(n_windows, n_10s)
    y_m_resh = y_m[:n_windows * n_10s].reshape(n_windows, n_10s)
    y_u_resh = y_u[:n_windows * n_10s].reshape(n_windows, n_10s)

    P_m = np.mean(prob_m_resh, axis=1)
    P_u = np.mean(prob_u_resh, axis=1)
    acc_m = np.mean((prob_m_resh > 0.5) == y_m_resh, axis=1)
    acc_u = np.mean((prob_u_resh > 0.5) == y_u_resh, axis=1)

    window_labels = np.array([
        1 if np.mean(s_total[i * win_len_samples:(i + 1) * win_len_samples]) > 0.5 else 0
        for i in range(n_windows)
    ])

    rng = np.random.RandomState(42)
    futures = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for win_idx in range(n_windows):
            lbl = window_labels[win_idx]
            future = executor.submit(_vision_sample_task,
                                     win_idx, lbl, occupied_pool, empty_pool,
                                     v_gamma, rng)
            futures.append(future)
        win_indices = np.empty(n_windows, dtype=int)
        degraded_images = [None] * n_windows
        brightness_arr = np.empty(n_windows) if return_brightness else None
        for future in tqdm(as_completed(futures), total=n_windows, desc='视觉退化'):
            wid, img_np, bright = future.result()
            win_indices[wid] = wid
            degraded_images[wid] = img_np
            if return_brightness:
                brightness_arr[wid] = bright
    sorted_idx = np.argsort(win_indices)
    degraded_images = [degraded_images[i] for i in sorted_idx]
    if return_brightness:
        brightness_arr = brightness_arr[sorted_idx]

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    P_v = _batch_predict(det_v, degraded_images, eval_transform, DEVICE, batch_size=128)
    acc_v = ((P_v > 0.5).astype(int) == window_labels).astype(float)

    if return_brightness:
        return P_v, P_m, P_u, window_labels, acc_v, acc_m, acc_u, brightness_arr
    else:
        return P_v, P_m, P_u, window_labels, acc_v, acc_m, acc_u


# ==================== 退化等级参数 ====================
# 视觉：gamma  地磁：emi   超声：multipath_p
SEVERITY_PARAMS = {
    0: {'v_gamma': 1.0, 'emi': 0, 'multipath_p': 0.0},
    1: {'v_gamma': 0.3, 'emi': 1.5, 'multipath_p': 0.25},
    2: {'v_gamma': 0.25, 'emi': 2, 'multipath_p': 0.5},
    3: {'v_gamma': 0.2, 'emi': 4, 'multipath_p': 1},
    4: {'v_gamma': 0.15, 'emi': 5, 'multipath_p': 2},
}


def generate_signals_by_scene_and_level(scene, level, s_total):
    """根据场景和退化等级生成地磁与超声信号"""
    params = SEVERITY_PARAMS[level]
    cfg = {
        'v_gamma': None,
        'emi': 0,
        'multipath_p': 0.0
    }
    if 'v' in scene:
        cfg['v_gamma'] = params['v_gamma']
    if 'm' in scene:
        cfg['emi'] = params['emi']
    if 'u' in scene:
        cfg['multipath_p'] = params['multipath_p']

    seed_offset = hash(scene) % 10
    # 地磁
    mag_sim = GeomagSimulator(seed=11451 + seed_offset)
    B_clean = mag_sim.generate_baseline(168 * 3600)
    B_clean = mag_sim.inject_self_occupancy(B_clean, s_total)
    if cfg['emi'] > 0:
        B_signal = mag_sim.inject_em_interference(B_clean, A_total=cfg['emi'])
    else:
        B_signal = B_clean

    # 超声
    us_sim = UltrasonicSimulator(seed=43 + seed_offset)
    d_clean = us_sim.generate_base_distance(s_total)
    if cfg['multipath_p'] > 0:
        d_signal = us_sim.inject_multipath(d_clean, severity=cfg['multipath_p'])
        d_signal = us_sim.inject_white_noise(d_signal)
    else:
        d_signal = us_sim.inject_white_noise(d_clean)

    return B_signal, d_signal, cfg


# ==================== 主函数 ====================
def main():
    print("H4：多退化等级融合算法对比")

    s_total = generate_event_sequence(168, weekday=True, seed=42)
    print(f"占用序列长度: {len(s_total):,}")

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

    win_len_samples = WINDOW_SEC * FS  # 6000
    scenes = ['v', 'm', 'u', 'vm', 'vu', 'mu', 'vmu']
    # scenes = ['u']
    levels = [0, 1, 2, 3, 4]
    # levels = [ 1, 2, 3, 4]
    # 清空旧的汇总文件（如需保留历史，可删除下面两行）
    summary_csv_path = os.path.join(RESULTS_DIR, 'h4_results.csv')
    if os.path.exists(summary_csv_path):
        os.remove(summary_csv_path)

    for level in levels:
        print(f"\n====== 退化等级 {level} ======")
        for scene in scenes:
            print(f"  场景: {scene}")
            B_signal, d_signal, cfg = generate_signals_by_scene_and_level(scene, level, s_total)

            v_gamma = cfg['v_gamma'] if cfg['v_gamma'] is not None else 1.0
            current_multipath = cfg['multipath_p']
            current_emi = cfg['emi']

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

            # 获取窗口概率（亮度用于 Ours）
            P_v, P_m, P_u, labels, acc_v, acc_m, acc_u, brightness = collect_window_data(
                s_total, prob_m, y_m, prob_u, y_u,
                det_v, win_len_samples,
                v_gamma=v_gamma,
                occupied_pool=occupied_pool, empty_pool=empty_pool,
                return_brightness=True
            )

            # 融合方法
            acc_mv = (majority_vote(P_v, P_m, P_u) == labels).astype(float)
            acc_ds_eq = (ds_fusion_fixed_weights(P_v, P_m, P_u, 1 / 3, 1 / 3, 1 / 3) == labels).astype(float)
            acc_ds_static = (ds_fusion_fixed_weights(P_v, P_m, P_u, 0.45, 0.30, 0.25) == labels).astype(float)
            w_v, w_u, w_m = compute_channel_weights(brightness, current_multipath, current_emi)
            fused_pred_ours = ds_fusion_decision_batch(P_v, P_m, P_u, w_v, w_u, w_m)
            acc_ours = (fused_pred_ours == labels).astype(float)

            # ==================== 窗口级 CSV ====================
            window_csv = os.path.join(RESULTS_DIR, f'h4_scene_{scene}_lev{level}_window.csv')
            with open(window_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['win_idx', 'label',
                                 'acc_v', 'acc_m', 'acc_u',
                                 'acc_mv', 'acc_ds_eq', 'acc_ds_static', 'acc_ours',
                                 'P_v', 'P_m', 'P_u',
                                 'w_v', 'w_m', 'w_u'])
                for i in range(len(labels)):
                    writer.writerow([i, labels[i],
                                     acc_v[i], acc_m[i], acc_u[i],
                                     acc_mv[i], acc_ds_eq[i], acc_ds_static[i], acc_ours[i],
                                     P_v[i], P_m[i], P_u[i],
                                     w_v[i], w_m[i], w_u[i]])

            # 汇总统计
            row = {
                'level': level,
                'scene': scene,
                'acc_v_mean': np.mean(acc_v),
                'acc_m_mean': np.mean(acc_m),
                'acc_u_mean': np.mean(acc_u),
                'acc_mv_mean': np.mean(acc_mv),
                'acc_ds_eq_mean': np.mean(acc_ds_eq),
                'acc_ds_static_mean': np.mean(acc_ds_static),
                'acc_ours_mean': np.mean(acc_ours),
            }
            row['best_single'] = max(row['acc_v_mean'], row['acc_m_mean'], row['acc_u_mean'])
            row['gain_mv_pp'] = (row['acc_mv_mean'] - row['best_single']) * 100
            row['gain_ds_eq_pp'] = (row['acc_ds_eq_mean'] - row['best_single']) * 100
            row['gain_ds_static_pp'] = (row['acc_ds_static_mean'] - row['best_single']) * 100
            row['gain_ours_pp'] = (row['acc_ours_mean'] - row['best_single']) * 100

            # 写入汇总 CSV
            fieldnames = list(row.keys())
            write_header = not os.path.exists(summary_csv_path)
            with open(summary_csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)

            print(f"    V:{row['acc_v_mean']:.3f} M:{row['acc_m_mean']:.3f} U:{row['acc_u_mean']:.3f} | "
                  f"MV:{row['acc_mv_mean']:.3f} DSEq:{row['acc_ds_eq_mean']:.3f} DSSt:{row['acc_ds_static_mean']:.3f} "
                  f"Ours:{row['acc_ours_mean']:.3f}")

    print(f"\nH4 汇总结果已实时保存至 {summary_csv_path}")


if __name__ == "__main__":
    main()
