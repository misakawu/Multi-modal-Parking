# run_h2_v2.py
"""
阶段④ H2 独立性验证 + Oracle 上界（融合版：3+3+1 故障场景）

实验设计：
- 固定中等退化水平（severity=2 参数）
- 7 个场景：单通道失效(3)、双通道失效(3)、全通道失效(1)
- 每个场景独立生成通道信号，未失效通道保持正常（零退化）
- 使用环境自适应加权 D‑S 融合计算融合准确率
- 输出场景汇总 CSV 和各场景窗口级 CSV
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
    """images_np: list of uint8 numpy arrays (H,W,3)"""
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


# ==================== 通道权重计算 ====================
def compute_channel_weights(brightness_arr, multipath_p, emi):
    # 视觉：根据 PKLot 正常图像亮度统计，半衰点 0.55，斜率 5
    k_v, b_v = 15.0, 0.2
    # 超声：多径概率 p 达到 0.6 时权重应接近 0，半衰点 0.45 保持，斜率适当加大至 10
    k_u, b_u = 10.0, 0.45
    # 地磁：EMI=20 时权重应接近 0，半衰点 10 不变，斜率 0.5 在 20 时权重约 0.007
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


def ds_fusion_decision_batch(P_v, P_m, P_u, w_v, w_u, w_m):
    n = len(P_v)
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
    m_avg = np.column_stack([
        (m_v[:, 0] * w_v + m_m[:, 0] * w_m + m_u[:, 0] * w_u) / w_total_safe,
        (m_v[:, 1] * w_v + m_m[:, 1] * w_m + m_u[:, 1] * w_u) / w_total_safe,
        (m_v[:, 2] * w_v + m_m[:, 2] * w_m + m_u[:, 2] * w_u) / w_total_safe
    ])
    m_temp = ds_combine(m_avg, m_avg)
    m_final_high = ds_combine(m_temp, m_avg)
    low_conflict = K_avg < 0.7
    H1 = np.where(low_conflict, m_final_low[:, 0], m_final_high[:, 0])
    H2 = np.where(low_conflict, m_final_low[:, 1], m_final_high[:, 1])
    pred = np.full(n, -1, dtype=int)
    pred[H1 > 0.6] = 1
    pred[H2 > 0.6] = 0
    return pred


# ==================== 滚动窗口分析 ====================
def rolling_window_analysis(
        s_total, prob_m, y_m, prob_u, y_u,
        det_v, win_len_samples, output_csv,
        v_gamma, v_noise_sigma, multipath_p, emi_amplitude,
        occupied_pool, empty_pool
):
    n_windows = len(s_total) // win_len_samples
    n_10s = win_len_samples // 100

    # 地磁/超声 10秒窗口聚合成 10分钟窗口
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

    # 多线程视觉退化
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
        brightness_arr = np.empty(n_windows)
        for future in tqdm(as_completed(futures), total=n_windows, desc='视觉退化'):
            wid, img_np, bright = future.result()
            win_indices[wid] = wid
            degraded_images[wid] = img_np
            brightness_arr[wid] = bright
    sorted_idx = np.argsort(win_indices)
    degraded_images = [degraded_images[i] for i in sorted_idx]
    brightness_arr = brightness_arr[sorted_idx]

    # GPU批量推理
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    P_v = _batch_predict(det_v, degraded_images, eval_transform, DEVICE, batch_size=128)
    acc_v = ((P_v > 0.5).astype(int) == window_labels).astype(float)

    # 权重与融合
    w_v, w_u, w_m = compute_channel_weights(brightness_arr, multipath_p, emi_amplitude)
    fused_pred = ds_fusion_decision_batch(P_v, P_m, P_u, w_v, w_u, w_m)
    acc_fused = (fused_pred == window_labels).astype(float)

    # 写入窗口CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['win_idx', 'label', 'acc_v', 'acc_m', 'acc_u', 'acc_fused',
                         'P_v', 'P_m', 'P_u', 'brightness', 'multipath_p', 'emi',
                         'w_v', 'w_m', 'w_u'])
        for i in range(n_windows):
            writer.writerow([i, window_labels[i],
                             acc_v[i], acc_m[i], acc_u[i], acc_fused[i],
                             P_v[i], P_m[i], P_u[i],
                             brightness_arr[i], multipath_p, emi_amplitude,
                             w_v[i], w_m[i], w_u[i]])
    return acc_v, acc_m, acc_u, acc_fused


# ==================== 获取重度退化参数 ====================
# SEVERITY_CONFIG = {
#     'cv': {
#         'gamma': [1.0, 0.7, 0.5, 0.3, 0.2],
#         'noise': [0, 10, 20, 30, 40]
#     },
#     'us': {
#         'multipath': [0, 0.15, 0.3, 0.45, 0.6],
#         'condensation': [1, 2, 3, 4, 5]
#     },
#     'mag': {
#         'em': [0, 5, 10, 20, 30],
#         'drift': [1, 3, 10, 30, 100]
#     }
# }
def get_mild_params():
    return {'v_gamma': 0.6, 'v_noise_sigma': 20,
            'emi': 20, 'drift': 10,
            'multipath_p': 0.5, 'condensation_sev': 4}


# ==================== 根据场景控制通道退化 ====================
def generate_signals_by_scene(scene, s_total):
    # scene: 例如 'v' 表示只有视觉失效，'vu' 表示视觉和超声失效
    p = get_mild_params()
    # 默认正常参数（零退化）
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
    # 生成地磁
    mag_sim = GeomagSimulator(seed=11451 + seed_offset)
    B_clean = mag_sim.generate_baseline(168 * 3600)
    B_clean = mag_sim.inject_self_occupancy(B_clean, s_total)
    if cfg['emi'] > 0:
        B_signal = mag_sim.inject_em_interference(B_clean, A_total=cfg['emi'])
        B_signal = mag_sim.inject_drift(B_signal, severe=cfg['drift'])
    else:
        B_signal = B_clean

    # 生成超声
    us_sim = UltrasonicSimulator(seed=43 + seed_offset)
    d_clean = us_sim.generate_base_distance(s_total)
    if cfg['multipath_p'] > 0:
        d_signal = us_sim.inject_multipath(d_clean, p=cfg['multipath_p'])
        d_signal = us_sim.inject_white_noise(d_signal)
        d_signal = us_sim.inject_condensation(d_signal, severity=cfg['condensation_sev'])
    else:
        d_signal = us_sim.inject_white_noise(d_clean)  # 只加正常噪声

    return B_signal, d_signal, cfg


# ==================== 主函数 ====================
def main():
    print("=" * 70)
    print("H2 实验：3+3+1 故障场景测试")
    print("=" * 70)

    # 1. 占用真值 (1周)
    s_total = generate_event_sequence(168, weekday=True, seed=42)
    print(f"占用序列长度: {len(s_total):,}")

    # 2. 加载检测器
    det_v = VisualDetector(CV_MODEL_PATH)
    with open('models/geomag_svm_model.pkl', 'rb') as f:
        det_m = GeomagDetector()
        det_m.svm = pickle.load(f)
    with open('models/us_params.json', 'r') as f:
        us_params = json.load(f)
    det_u = UltrasonicDetector(d_threshold=us_params['d_threshold'],
                               alpha=us_params['alpha'])

    # 3. 加载视觉图片池（只加载一次）
    print("加载视觉图片池...")
    occupied_pool, empty_pool = load_vision_pools(PUC_INDEX_FILE, max_samples=2000)

    # 4. 定义7个场景
    scenes = ['v', 'm', 'u', 'vm', 'vu', 'mu', 'vmu']  # 单通道失效 + 双通道失效 + 全失效
    # scenes = [ 'mu', 'vmu']
    win_len_samples = WINDOW_SEC * FS  # 6000

    all_summaries = []

    for scene in scenes:
        print(f"\n{'=' * 20} 场景: {scene} {'=' * 20}")
        # 生成信号（根据scene控制退化）
        B_signal, d_signal, cfg = generate_signals_by_scene(scene, s_total)

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

        # 提取环境参数用于权重（注意：超声多径概率和地磁EMI已根据场景设置）
        current_multipath = cfg['multipath_p']
        current_emi = cfg['emi']

        # 视觉退化参数（可能为None，None时传入1.0和0）
        v_gamma = cfg['v_gamma'] if cfg['v_gamma'] is not None else 1.0
        v_noise = cfg['v_noise_sigma'] if cfg['v_noise_sigma'] is not None else 0.0

        # 滚动窗口分析
        window_csv = os.path.join(RESULTS_DIR, f'h2_scene_{scene}_window.csv')
        acc_v, acc_m, acc_u, acc_fused = rolling_window_analysis(
            s_total, prob_m, y_m, prob_u, y_u,
            det_v, win_len_samples, window_csv,
            v_gamma=v_gamma, v_noise_sigma=v_noise,
            multipath_p=current_multipath, emi_amplitude=current_emi,
            occupied_pool=occupied_pool, empty_pool=empty_pool
        )

        # 汇总统计
        summary = {
            'scene': scene,
            'acc_v_mean': np.mean(acc_v),
            'acc_m_mean': np.mean(acc_m),
            'acc_u_mean': np.mean(acc_u),
            'acc_fused_mean': np.mean(acc_fused),
            'best_single': max(np.mean(acc_v), np.mean(acc_m), np.mean(acc_u)),
            'oracle_gain_pp': (np.mean(acc_fused) - max(np.mean(acc_v), np.mean(acc_m), np.mean(acc_u))) * 100
        }
        print(f"  视觉: {summary['acc_v_mean']:.3f}, 地磁: {summary['acc_m_mean']:.3f}, "
              f"超声: {summary['acc_u_mean']:.3f}, 融合: {summary['acc_fused_mean']:.3f}, "
              f"增益: {summary['oracle_gain_pp']:.2f} pp")
        all_summaries.append(summary)

    # 保存汇总
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(os.path.join(RESULTS_DIR, 'h2_scene_summary.csv'), index=False)
    print("\n场景汇总已保存至 h2_scene_summary.csv")


if __name__ == "__main__":
    main()
