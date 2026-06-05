# run_h1.py
import csv
import json
import pickle

import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torchvision import transforms
from tqdm import tqdm

from config import *
from detectors import VisualDetector, UltrasonicDetector, GeomagDetector
from feature_engineering import compute_clean_background, extract_geomag_features_clean
from sim_events import generate_event_sequence
from sim_geomag import GeomagSimulator
from sim_ultrasonic import UltrasonicSimulator

# ----------------------------------------------------------------------
# 严重度档位定义
# ----------------------------------------------------------------------
SEVERITY_CONFIG = {
    'cv': {
        'gamma': [1.0, 0.7, 0.5, 0.3, 0.2],
        'noise': [0, 10, 20, 30, 40],
        # 'oil_dirt': [0, 0.10, 0.20, 0.30, 0.40]
    },
    'us': {
        'multipath': [0, 0.15, 0.3, 0.45, 0.6],
        # 'condensation': [1, 2, 3, 4, 5],
        # 'packet_loss': [0, 0.10, 0.20, 0.35, 0.50]
    },
    'mag': {
        'em': [0, 1.5, 2, 4, 6],
        # 'crosstalk': [0, 1, 2, 3, 4,5,6],
        'drift': [1, 3, 10, 30, 100]
    }
}

FIELD_NAMES = ['channel', 'failure', 'severity', 'accuracy', 'f1']

H1_SEED = 1154


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------
def append_row(csv_path, row_dict, write_header=False):
    mode = 'w' if write_header else 'a'
    with open(csv_path, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELD_NAMES)
        if write_header:
            writer.writeheader()
        if row_dict is not None:
            writer.writerow(row_dict)


def make_geomag_test_set(failure_mode, severity, n_samples=2000):
    sim = GeomagSimulator(seed=H1_SEED)
    dur = 24.0
    s = generate_event_sequence(dur, seed=H1_SEED + 1)
    B = sim.generate_baseline(dur * 3600)
    B = sim.inject_self_occupancy(B, s)
    if failure_mode == 'em':
        if severity > 0:
            B = sim.inject_em_interference(B, A_total=severity)
    elif failure_mode == 'crosstalk':
        if severity > 0:
            # 完整邻位配置：4个横向 + 2个纵向
            nb_dists = [2.5, 2.5, 5.0, 5.0, 5.5, 5.5]  # 左1,右1,左2,右2,前对,后对
            n_neighbors = min(severity, len(nb_dists))
            nbs = [generate_event_sequence(dur, seed=H1_SEED + 2 + i) for i in range(n_neighbors)]
            neighbors = [{'r': nb_dists[i], 's': nbs[i]} for i in range(n_neighbors)]
            B = sim.inject_neighbor_crosstalk(B, neighbors)
    elif failure_mode == 'drift':
        B = sim.inject_drift(B, severe=severity)
    B_bg = compute_clean_background(B, s)
    X, y = extract_geomag_features_clean(B, s, B_bg)
    idx1 = np.where(y == 1)[0]
    idx0 = np.where(y == 0)[0]
    n = min(len(idx1), len(idx0), n_samples // 2)
    idx = np.concatenate([np.random.choice(idx1, n, replace=False),
                          np.random.choice(idx0, n, replace=False)])
    return X[idx], y[idx]


def make_ultrasonic_test_set(failure_mode, severity, n_samples=5000):
    sim = UltrasonicSimulator(seed=H1_SEED + 3)
    dur = 24.0
    s = generate_event_sequence(dur, seed=H1_SEED + 4)
    d_true = sim.generate_base_distance(s)

    if failure_mode == 'multipath':
        d = sim.inject_white_noise(d_true)
        d = sim.inject_multipath(d, severity)
    elif failure_mode == 'condensation':
        d = sim.inject_condensation(d_true, severity)
    elif failure_mode == 'packet_loss':
        d = sim.inject_white_noise(d_true)
        d, _ = sim.inject_packet_loss(d, severity)
    else:
        d = sim.inject_white_noise(d_true)

    win_len = 10
    X, y = [], []
    for i in range(0, len(s) - win_len + 1, win_len):
        win_d = d[i:i + win_len]
        win_s = s[i:i + win_len]
        d_med = np.nanmedian(win_d)
        if np.isnan(d_med):
            d_med = H_CEILING
        X.append([d_med])
        y.append(1 if np.mean(win_s) > 0.5 else 0)
    X, y = np.array(X), np.array(y)
    idx1 = np.where(y == 1)[0]
    idx0 = np.where(y == 0)[0]
    n = min(len(idx1), len(idx0), n_samples // 2)
    idx = np.concatenate([np.random.choice(idx1, n, replace=False),
                          np.random.choice(idx0, n, replace=False)])
    return X[idx], y[idx]


@torch.no_grad()
def CV_evaluate(det_v, dataloader, failure, severity):
    # ---------- 视觉通道评估函数 ----------
    all_labels = []
    all_preds = []
    all_probs = []  # 占用类的概率
    det_v.model.eval()
    for images, labels in tqdm(dataloader, desc='Evaluating', leave=False):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = det_v.predict_proba(images)
        probs = F.softmax(outputs, dim=1)
        # 占用类索引为 1 (0:Empty, 1:Occupied)
        occ_probs = probs[:, 1].cpu().numpy()
        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(occ_probs)
    acc_v = np.mean(np.array(all_preds) == np.array(all_labels))
    f1_v = f1_score(all_labels, all_preds, zero_division=0)
    append_row(csv_path, {'channel': 'cv', 'failure': failure, 'severity': severity,
                          'accuracy': acc_v, 'f1': f1_v})
    print(f"CV 退化： {failure} 严重程度： {severity} Acc={acc_v:.4f}, F1={f1_v:.4f}")


# 与训练完全相同的预处理（用于基准测试直接加载Tensor）
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, 'h1_results.csv')
    append_row(csv_path, None, write_header=True)

    # ---------- 加载预训练模型 ----------
    det_v = VisualDetector(CV_MODEL_PATH)
    with open('models/geomag_svm_model.pkl', 'rb') as f:
        svm_model = pickle.load(f)
    det_m = GeomagDetector()
    det_m.svm = svm_model
    with open('models/us_params.json', 'r') as f:
        us_params = json.load(f)
    det_u = UltrasonicDetector(d_threshold=us_params['d_threshold'],
                               alpha=us_params['alpha'])

    # # ----------- 视觉 H1 ---------------
    # print("===== 视觉 H1 =====")
    #
    # # 基准测试
    # base_ds = ParkingDataset(PUC_INDEX_FILE, transform=eval_transform)
    # base_loader = DataLoader(base_ds, batch_size=64, shuffle=False, num_workers=4)
    # CV_evaluate(det_v=det_v, dataloader=base_loader, failure='non', severity=0)
    #
    # # 退化测试：
    # for fail_type, params in SEVERITY_CONFIG['cv'].items():
    #     for sev, param in enumerate(params):
    #         if sev == 0: continue
    #
    #         ds = VisualTestDataset(PUC_INDEX_FILE, transform=eval_transform, degrade_type=fail_type,
    #                                degrade_param=param)
    #         loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
    #         CV_evaluate(det_v=det_v, dataloader=loader, failure=fail_type, severity=sev)
    #
    # ----------- 超声 H1 ---------------
    # print("===== 超声 H1 =====")
    # X_base_u, y_base_u = make_ultrasonic_test_set('none', 0)
    # pred_base = (det_u.predict_proba(X_base_u.flatten()) > 0.5).astype(int)
    # base_acc_u = accuracy_score(y_base_u, pred_base)
    # f1 = f1_score(y_base_u, pred_base, zero_division=0)
    # append_row(csv_path, {'channel': 'us', 'failure': 'none', 'severity': 0,
    #                       'accuracy': base_acc_u, 'f1': f1})
    # print(f"US 无退化 Acc={base_acc_u:.4f} f1={f1:.4f}")
    #
    # for fail_type, params in SEVERITY_CONFIG['us'].items():
    #     for sev, param in enumerate(params):
    #         if sev == 0: continue
    #         X, y = make_ultrasonic_test_set(fail_type, param)
    #         prob = det_u.predict_proba(X.flatten())
    #         pred = (prob > 0.5).astype(int)
    #         acc = accuracy_score(y, pred)
    #         f1 = f1_score(y, pred, zero_division=0)
    #         append_row(csv_path, {'channel': 'us', 'failure': fail_type, 'severity': sev,
    #                               'accuracy': acc, 'f1': f1})
    #         print(f"US {fail_type} ε={sev}  Acc={acc:.4f}, F1={f1:.4f}")

    # ----------- 地磁 H1 ---------------
    print("===== 地磁 H1 =====")
    X_base_m, y_base_m = make_geomag_test_set('none', 0)
    prob_base = det_m.predict_proba(X_base_m)
    pred_base_m = (prob_base > 0.5).astype(int)
    base_acc_m = accuracy_score(y_base_m, pred_base_m)
    f1 = f1_score(y_base_m, pred_base_m, zero_division=0)
    append_row(csv_path, {'channel': 'mag', 'failure': 'none', 'severity': 0,
                          'accuracy': base_acc_m, 'f1': f1})
    print(f"Mag 无退化 Acc={base_acc_m:.4f} f1={f1:.4f}")

    for fail_type, params in SEVERITY_CONFIG['mag'].items():
        for sev, param in enumerate(params):
            if sev == 0: continue
            X, y = make_geomag_test_set(fail_type, param)
            prob = det_m.predict_proba(X)
            pred = (prob > 0.5).astype(int)
            acc = accuracy_score(y, pred)
            f1 = f1_score(y, pred, zero_division=0)
            append_row(csv_path, {'channel': 'mag', 'failure': fail_type, 'severity': sev,
                                  'accuracy': acc, 'f1': f1})
            print(f"Mag {fail_type} ε={sev}  Acc={acc:.4f}, F1={f1:.4f}")

    print(f"\nH1 全部完成，结果保存至 {csv_path}")
