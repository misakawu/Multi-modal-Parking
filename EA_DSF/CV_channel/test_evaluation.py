import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models
from tqdm import tqdm

from parking_dataset import ParkingDataset

# 尝试导入 sklearn 计算指标
try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix
    )

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("警告：未安装 scikit-learn，将仅计算准确率，其他指标置为 -1。请执行 pip install scikit-learn")

# from parking_dataset_test import ParkingDataset  # 独立定义的数据集类

# -------------------- 配置 --------------------
PKLOT_INDEX_DIR = r"E:\DATASET\PKLot\index"
CNR_INDEX_FILE = r"E:\DATASET\CNR-EXT-Patches\index.json"
MODEL_WEIGHTS = "best_model.pth"
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 退化模拟参数
GAMMA_VALUES = [0.3, 0.5, 0.7]
NOISE_SIGMA_VALUES = [10, 20, 30]

OUTPUT_CSV = "CV_test_results.csv"


# -------------------- 模型构建 --------------------
def build_model(num_classes=2):
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


# -------------------- 图像预处理 --------------------
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# -------------------- 评估函数（返回完整指标） --------------------
@torch.no_grad()
def evaluate_extended(model, dataloader, device):
    """
    评估模型并返回多个指标。
    返回字典包含：size, accuracy, precision, recall, f1, auc, confusion_matrix
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []  # 占用类的概率

    for images, labels in tqdm(dataloader, desc='Evaluating', leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        # 占用类索引为 1 (0:Empty, 1:Occupied)
        occ_probs = probs[:, 1].cpu().numpy()
        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(occ_probs)

    total = len(all_labels)
    if total == 0:
        return {'size': 0, 'accuracy': 0, 'precision': -1, 'recall': -1,
                'f1': -1, 'auc': -1, 'confusion_matrix': 'N/A'}

    # 准确率（无论sklearn可用与否都可计算）
    acc = np.mean(np.array(all_preds) == np.array(all_labels))

    if SKLEARN_AVAILABLE:
        try:
            precision = precision_score(all_labels, all_preds, zero_division=0)
            recall = recall_score(all_labels, all_preds, zero_division=0)
            f1 = f1_score(all_labels, all_preds, zero_division=0)
            # AUC 需要至少两个类别都存在
            unique_labels = np.unique(all_labels)
            if len(unique_labels) == 2:
                auc = roc_auc_score(all_labels, all_probs)
            else:
                auc = float('nan')
            cm = confusion_matrix(all_labels, all_preds)
            cm_str = str(cm.tolist())  # 例如 "[[TN, FP], [FN, TP]]"
        except Exception as e:
            print(f"计算指标时出错: {e}")
            precision = recall = f1 = auc = -1
            cm_str = 'error'
    else:
        precision = recall = f1 = auc = -1
        cm_str = 'N/A'

    return {
        'size': total,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm_str
    }


# -------------------- CSV 即时保存函数 --------------------
def save_result(csv_path, result_dict, write_header=False):
    """将单条结果写入 CSV，支持即时追加"""
    mode = 'w' if write_header else 'a'
    fieldnames = [
        'test_set', 'degrade_type', 'degrade_param',
        'size', 'accuracy', 'precision', 'recall', 'f1', 'auc', 'confusion_matrix'
    ]
    with open(csv_path, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(result_dict)


# -------------------- 主测试流程 --------------------
def main():
    print(f"使用设备: {DEVICE}")

    # 第一次写入时创建/覆盖 CSV 并写表头
    first_write = True
    csv_path = OUTPUT_CSV

    # 1. 加载模型
    print("加载模型权重...")
    model = build_model(num_classes=2).to(DEVICE)
    state_dict = torch.load(MODEL_WEIGHTS, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    # 2. 测试集 A：PKLot PUC
    print("\n========== 测试集 A: PKLot PUC ==========")
    puc_dataset = ParkingDataset(
        index_file=os.path.join(PKLOT_INDEX_DIR, "puc_test.json"),
        transform=eval_transform
    )
    puc_loader = DataLoader(puc_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    metrics = evaluate_extended(model, puc_loader, DEVICE)
    print(f"PUC 测试结果: Size={metrics['size']}, Acc={metrics['accuracy']:.4f}, "
          f"Prec={metrics['precision']:.4f}, Rec={metrics['recall']:.4f}, "
          f"F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}")
    save_result(csv_path, {
        'test_set': 'PKLot PUC',
        'degrade_type': 'None',
        'degrade_param': '',
        'size': metrics['size'],
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'auc': metrics['auc'],
        'confusion_matrix': metrics['confusion_matrix']
    }, write_header=first_write)
    first_write = False

    # 3. 测试集 B：CNR-EXT
    print("\n========== 测试集 B: CNR-EXT-Patches ==========")
    if os.path.exists(CNR_INDEX_FILE):
        cnr_dataset = ParkingDataset(
            index_file=CNR_INDEX_FILE,
            transform=eval_transform
        )
        cnr_loader = DataLoader(cnr_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        metrics = evaluate_extended(model, cnr_loader, DEVICE)
        print(f"CNR-EXT 测试结果: Size={metrics['size']}, Acc={metrics['accuracy']:.4f}, "
              f"Prec={metrics['precision']:.4f}, Rec={metrics['recall']:.4f}, "
              f"F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}")
        save_result(csv_path, {
            'test_set': 'CNR-EXT',
            'degrade_type': 'None',
            'degrade_param': '',
            'size': metrics['size'],
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'auc': metrics['auc'],
            'confusion_matrix': metrics['confusion_matrix']
        })
    else:
        print(f"未找到 CNR-EXT 索引文件: {CNR_INDEX_FILE}")

    # 4. 退化模拟：亮度衰减
    print("\n========== 退化模拟: 亮度衰减 ==========")
    for gamma in GAMMA_VALUES:
        degrade_dataset = ParkingDataset(
            index_file=os.path.join(PKLOT_INDEX_DIR, "puc_test.json"),
            transform=eval_transform,
            degrade_gamma=gamma
        )
        degrade_loader = DataLoader(degrade_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        metrics = evaluate_extended(model, degrade_loader, DEVICE)
        print(f"Gamma={gamma} | Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}")
        save_result(csv_path, {
            'test_set': 'PKLot PUC',
            'degrade_type': 'gamma',
            'degrade_param': str(gamma),
            'size': metrics['size'],
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'auc': metrics['auc'],
            'confusion_matrix': metrics['confusion_matrix']
        })

    # 5. 退化模拟：高斯噪声
    print("\n========== 退化模拟: 高斯噪声 ==========")
    for sigma in NOISE_SIGMA_VALUES:
        degrade_dataset = ParkingDataset(
            index_file=os.path.join(PKLOT_INDEX_DIR, "puc_test.json"),
            transform=eval_transform,
            degrade_noise_sigma=sigma
        )
        degrade_loader = DataLoader(degrade_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        metrics = evaluate_extended(model, degrade_loader, DEVICE)
        print(f"Sigma={sigma} | Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}")
        save_result(csv_path, {
            'test_set': 'PKLot PUC',
            'degrade_type': 'noise_sigma',
            'degrade_param': str(sigma),
            'size': metrics['size'],
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'auc': metrics['auc'],
            'confusion_matrix': metrics['confusion_matrix']
        })

    print(f"\n所有测试完成，详细结果已保存至 {csv_path}。")


if __name__ == '__main__':
    main()
