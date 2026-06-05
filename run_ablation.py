# run_ablation.py
"""
消融实验：Ours 融合算法组件分析（基于实际 H4 实现）

变体：
  Full        自适应权重 + 冲突自适应 (当前 H4 的 Ours)
  -Weight     等权重 (1/3) + 冲突自适应
  -Conflict   自适应权重 + 标准 D-S (无冲突处理)
  -Adapt      等权重 + 标准 D-S (即 DS-Equal)

完全复用 run_h4.py 的辅助函数和仿真逻辑，仅融合部分按变体调整。
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


# ==================== 从 run_h4.py 原样复制辅助函数 ====================
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
    degraded = degrade_lowlight(img_np, v_gamma)  # 注意：此处调用与 H4 完全一致
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


# 注意：权重计算函数已包含 min_weight=0.1 的保护
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

    min_weight = 0.1
    w_v = np.maximum(w_v, min_weight)
    w_m = np.maximum(w_m, min_weight)
    w_u = np.maximum(w_u, min_weight)
    return w_v, w_u, w_m


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
    """标准 D-S 组合（无冲突检测），动态阈值与 H4 完全相同"""
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
    Theta = m_final[:, 2]
    decision_threshold = 0.55 + 0.15 * Theta
    pred = np.full(n, -1, dtype=int)
    pred[H1 > decision_threshold] = 1
    pred[H2 > decision_threshold] = 0
    return pred


def ds_fusion_decision_batch(P_v, P_m, P_u, w_v, w_u, w_m):
    """冲突自适应 D-S (Ours)，与 H4 完全相同"""
    n = len(P_v)

    # 将标量权重转换为数组
    if np.isscalar(w_v):
        w_v = np.full(n, w_v)
    if np.isscalar(w_m):
        w_m = np.full(n, w_m)
    if np.isscalar(w_u):
        w_u = np.full(n, w_u)

    m_v = np.column_stack([P_v * w_v, (1 - P_v) * w_v, 1 - w_v])
    m_m = np.column_stack([P_m * w_m, (1 - P_m) * w_m, 1 - w_m])
    m_u = np.column_stack([P_u * w_u, (1 - P_u) * w_u, 1 - w_u])

    K_vm = m_v[:, 0] * m_m[:, 1] + m_v[:, 1] * m_m[:, 0]
    K_vu = m_v[:, 0] * m_u[:, 1] + m_v[:, 1] * m_u[:, 0]
    K_mu = m_m[:, 0] * m_u[:, 1] + m_m[:, 1] * m_u[:, 0]
    K_avg = (K_vm + K_vu + K_mu) / 3.0

    m12 = ds_combine(m_v, m_m)
    m_final_low = ds_combine(m12, m_u)

    w_total = w_v + w_m + w_u
    mask_zero = w_total == 0
    w_total_safe = np.where(mask_zero, 1.0, w_total)

    w_v_norm = w_v / w_total_safe
    w_m_norm = w_m / w_total_safe
    w_u_norm = w_u / w_total_safe

    max_weights = np.maximum.reduce([w_v_norm, w_m_norm, w_u_norm])

    w_v_enhanced = np.where(w_v_norm == max_weights, w_v_norm * 1.5, w_v_norm)
    w_m_enhanced = np.where(w_m_norm == max_weights, w_m_norm * 1.5, w_m_norm)
    w_u_enhanced = np.where(w_u_norm == max_weights, w_u_norm * 1.5, w_u_norm)

    w_sum_enhanced = w_v_enhanced + w_m_enhanced + w_u_enhanced
    w_v_enhanced /= w_sum_enhanced
    w_m_enhanced /= w_sum_enhanced
    w_u_enhanced /= w_sum_enhanced

    m_avg = np.column_stack([
        P_v * w_v_enhanced + P_m * w_m_enhanced + P_u * w_u_enhanced,
        (1 - P_v) * w_v_enhanced + (1 - P_m) * w_m_enhanced + (1 - P_u) * w_u_enhanced,
        1 - max_weights
    ])
    m_temp = ds_combine(m_avg, m_avg)
    m_final_high = ds_combine(m_temp, m_avg)

    adaptive_conflict_threshold = 0.5 + 0.3 * np.exp(-0.5 * w_total)
    low_conflict = K_avg < adaptive_conflict_threshold

    H1 = np.where(low_conflict, m_final_low[:, 0], m_final_high[:, 0])
    H2 = np.where(low_conflict, m_final_low[:, 1], m_final_high[:, 1])

    Theta_low = m_final_low[:, 2]
    Theta_high = m_final_high[:, 2]
    Theta = np.where(low_conflict, Theta_low, Theta_high)

    uncertainty_penalty = 0.1 * Theta
    conflict_bonus = 0.15 * (1 - np.exp(-2 * K_avg))
    confidence_gap = np.abs(H1 - H2)
    confidence_bonus = 0.1 * np.exp(-2 * confidence_gap)

    decision_threshold = 0.45 + uncertainty_penalty - conflict_bonus + confidence_bonus
    decision_threshold = np.clip(decision_threshold, 0.40, 0.65)

    pred = np.full(n, -1, dtype=int)
    pred[H1 > decision_threshold] = 1
    pred[H2 > decision_threshold] = 0

    # 后处理：多数投票兜底
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


def collect_window_data(
        s_total, prob_m, y_m, prob_u, y_u,
        det_v, win_len_samples,
        v_gamma,
        occupied_pool, empty_pool,
        return_brightness=False
):
    # 与 H4 保持一致，注意不再传递 v_noise_sigma
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


# 复制 H4 的退化参数（5个等级）
SEVERITY_PARAMS = {
    0: {'v_gamma': 1.0, 'emi': 0, 'multipath_p': 0.0},
    1: {'v_gamma': 0.3, 'emi': 1.5, 'multipath_p': 0.25},
    2: {'v_gamma': 0.25, 'emi': 2, 'multipath_p': 0.5},
    3: {'v_gamma': 0.2, 'emi': 4, 'multipath_p': 1},
    4: {'v_gamma': 0.15, 'emi': 5, 'multipath_p': 2},
}


def generate_signals_by_scene_and_level(scene, level, s_total):
    params = SEVERITY_PARAMS[level]
    cfg = {'v_gamma': None, 'emi': 0, 'multipath_p': 0.0}
    if 'v' in scene:
        cfg['v_gamma'] = params['v_gamma']
    if 'm' in scene:
        cfg['emi'] = params['emi']
    if 'u' in scene:
        cfg['multipath_p'] = params['multipath_p']

    seed_offset = hash(scene) % 10
    mag_sim = GeomagSimulator(seed=11451 + seed_offset)
    B_clean = mag_sim.generate_baseline(168 * 3600)
    B_clean = mag_sim.inject_self_occupancy(B_clean, s_total)
    if cfg['emi'] > 0:
        B_signal = mag_sim.inject_em_interference(B_clean, A_total=cfg['emi'])
    else:
        B_signal = B_clean

    us_sim = UltrasonicSimulator(seed=43 + seed_offset)
    d_clean = us_sim.generate_base_distance(s_total)
    if cfg['multipath_p'] > 0:
        d_signal = us_sim.inject_multipath(d_clean, severity=cfg['multipath_p'])
        d_signal = us_sim.inject_white_noise(d_signal)
    else:
        d_signal = us_sim.inject_white_noise(d_clean)

    return B_signal, d_signal, cfg


# ================== 消融实验主程序 ==================
def main():
    print("消融实验：Ours 组件分析（基于实际 H4 逻辑）")

    s_total = generate_event_sequence(168, weekday=True, seed=42)
    det_v = VisualDetector(CV_MODEL_PATH)
    with open('models/geomag_svm_model.pkl', 'rb') as f:
        det_m = GeomagDetector();
        det_m.svm = pickle.load(f)
    with open('models/us_params.json', 'r') as f:
        us_params = json.load(f)
    det_u = UltrasonicDetector(d_threshold=us_params['d_threshold'], alpha=us_params['alpha'])

    print("加载视觉图片池...")
    occupied_pool, empty_pool = load_vision_pools(PUC_INDEX_FILE, max_samples=2000)

    win_len_samples = WINDOW_SEC * FS
    # 与 H4 相同，使用全部 7 个场景和 5 个退化等级
    scenes = ['v', 'm', 'u', 'vm', 'vu', 'mu', 'vmu']
    levels = [0, 1, 2, 3, 4]
    variants = ['Full', '-Weight', '-Conflict', '-Adapt']

    summary_csv = os.path.join(RESULTS_DIR, 'ablation_results.csv')
    if os.path.exists(summary_csv):
        os.remove(summary_csv)

    for level in levels:
        print(f"\n===== 退化等级 {level} =====")
        for scene in scenes:
            print(f"  场景 {scene}: ", end='')
            B_signal, d_signal, cfg = generate_signals_by_scene_and_level(scene, level, s_total)

            v_gamma = cfg['v_gamma'] if cfg['v_gamma'] is not None else 1.0
            multipath = cfg['multipath_p']
            emi = cfg['emi']

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

            # 获取窗口概率与亮度（亮度仅用于 Full 和 -Conflict）
            P_v, P_m, P_u, labels, _, _, _, brightness = collect_window_data(
                s_total, prob_m, y_m, prob_u, y_u,
                det_v, win_len_samples, v_gamma,
                occupied_pool, empty_pool, return_brightness=True
            )

            # 计算自适应权重（用于 Full 和 -Conflict）
            w_v, w_u, w_m = compute_channel_weights(brightness, multipath, emi)

            accs = {}
            # Full: 自适应权重 + 冲突自适应
            pred = ds_fusion_decision_batch(P_v, P_m, P_u, w_v, w_u, w_m)
            accs['Full'] = np.mean(pred == labels)

            # -Weight: 等权重 + 冲突自适应
            pred_w = ds_fusion_decision_batch(P_v, P_m, P_u, 1 / 3, 1 / 3, 1 / 3)
            accs['-Weight'] = np.mean(pred_w == labels)

            # -Conflict: 自适应权重 + 标准 D-S (无冲突处理)
            pred_c = ds_fusion_fixed_weights(P_v, P_m, P_u, w_v, w_u, w_m)
            accs['-Conflict'] = np.mean(pred_c == labels)

            # -Adapt: 等权重 + 标准 D-S (即 DS-Equal)
            pred_a = ds_fusion_fixed_weights(P_v, P_m, P_u, 1 / 3, 1 / 3, 1 / 3)
            accs['-Adapt'] = np.mean(pred_a == labels)

            # 立即写入 CSV
            row = {'level': level, 'scene': scene}
            row.update(accs)
            write_header = not os.path.exists(summary_csv)
            with open(summary_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
            print(
                f"Full={accs['Full']:.3f}, -Weight={accs['-Weight']:.3f}, -Conflict={accs['-Conflict']:.3f}, -Adapt={accs['-Adapt']:.3f}")

    print(f"\n消融结果已保存至 {summary_csv}")


if __name__ == "__main__":
    main()
