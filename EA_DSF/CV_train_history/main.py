import torch
import torch.nn as nn
from mpmath.identification import transforms
from sympy.printing.pytorch import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from build_model import build_mobilenetv3_small
from cnrpark_ext import collect_cnrpark_ext
from parkingdataset import collect_data_paths, split_train_val_test, ParkingDataset
from train import train_model, evaluate


def main():
    # 参数配置
    data_root = r"E:\DATASET\PKLot\PKLot_organized"
    batch_size = 64
    epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 收集训练数据 (UFPR04 + UFPR05)
    print("--- 训练数据加载 ---")
    train_parking = ['UFPR04', 'UFPR05']
    paths, labels = collect_data_paths(data_root, train_parking)
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = \
        split_train_val_test(paths, labels)

    # 2. 数据预处理
    print("--- 数据预处理 ---")
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 改变图片大小
        transforms.RandomHorizontalFlip(),  # 随机翻转
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 均值和标准差
                             std=[0.229, 0.224, 0.225])
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 3. 创建 Dataset 和 DataLoader
    train_dataset = ParkingDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = ParkingDataset(val_paths, val_labels, transform=eval_transform)
    test_dataset = ParkingDataset(test_paths, test_labels, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 4. 构建模型
    model = build_mobilenetv3_small(num_classes=2, freeze_layers=True).to(device)

    # 5. 训练
    print("\n--- 训练 ---")
    train_model(model, train_loader, val_loader, epochs, device)

    # 6. 加载最佳模型并在测试集 A (PUC) 上评估
    model.load_state_dict(torch.load("../CV_channel/best_model.pth"))
    print("\n--- 测试集 A: PKLot PUC ---")
    puc_paths, puc_labels = collect_data_paths(data_root, ['PUC'])
    puc_dataset = ParkingDataset(puc_paths, puc_labels, transform=eval_transform)
    puc_loader = DataLoader(puc_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    criterion = nn.CrossEntropyLoss()
    puc_loss, puc_acc = evaluate(model, puc_loader, criterion, device)
    print(f"PUC Test Loss: {puc_loss:.4f}, Acc: {puc_acc:.4f}")


def run_full_evaluation():
    """
    对训练好的模型执行完整的测试评估，包括：
    - 测试集 A：PKLot PUC 停车场
    - 测试集 B：CNR-EXT-Patches 全部图像
    - 退化模拟（亮度衰减、高斯噪声）在 PUC 测试集上的表现
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_mobilenetv3_small(num_classes=2, freeze_layers=True).to(device)
    model.load_state_dict(torch.load("../CV_channel/best_model.pth"))
    data_root = r"E:\DATASET\PKLot\PKLot_organized"
    cnr_root = r"E:\DATASET\CNR-EXT-Patches"
    batch_size = 64

    criterion = nn.CrossEntropyLoss()
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ---------- 测试集 A：PKLot PUC ----------
    print("\n========== 测试集 A: PKLot PUC ==========")
    puc_paths, puc_labels = collect_data_paths(data_root, ['PUC'])
    puc_dataset = ParkingDataset(puc_paths, puc_labels, transform=eval_transform)
    puc_loader = DataLoader(puc_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    puc_loss, puc_acc = evaluate(model, puc_loader, criterion, device)
    print(f"PUC Test - Loss: {puc_loss:.4f}, Accuracy: {puc_acc:.4f}")

    # ---------- 测试集 B：CNR-EXT-Patches ----------
    print("\n========== 测试集 B: CNR-EXT-Patches ==========")
    cnr_paths, cnr_labels = collect_cnrpark_ext(cnr_root, label_file='all.txt')
    cnr_dataset = ParkingDataset(cnr_paths, cnr_labels, transform=eval_transform)
    cnr_loader = DataLoader(cnr_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    cnr_loss, cnr_acc = evaluate(model, cnr_loader, criterion, device)
    print(f"CNR-EXT-Patches Test - Loss: {cnr_loss:.4f}, Accuracy: {cnr_acc:.4f}")

    # ---------- 退化模拟：亮度衰减 (基于 PUC) ----------
    print("\n========== 退化模拟: 亮度衰减 ==========")
    for gamma in [0.3, 0.5, 0.7]:
        degrade_dataset = ParkingDataset(puc_paths, puc_labels, transform=eval_transform,
                                         degrade_gamma=gamma)
        degrade_loader = DataLoader(degrade_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        loss, acc = evaluate(model, degrade_loader, criterion, device)
        print(f"Gamma={gamma}  |  Accuracy: {acc:.4f}")

    # ---------- 退化模拟：高斯噪声 (基于 PUC) ----------
    print("\n========== 退化模拟: 高斯噪声 ==========")
    for sigma in [10, 20, 30]:
        degrade_dataset = ParkingDataset(puc_paths, puc_labels, transform=eval_transform,
                                         degrade_noise_sigma=sigma)
        degrade_loader = DataLoader(degrade_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        loss, acc = evaluate(model, degrade_loader, criterion, device)
        print(f"Sigma={sigma}  |  Accuracy: {acc:.4f}")


if __name__ == "__main__":
    # main()

    run_full_evaluation()
