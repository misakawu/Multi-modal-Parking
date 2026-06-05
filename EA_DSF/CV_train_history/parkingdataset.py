from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class ParkingDataset(Dataset):
    """停车位占用分类数据集，支持退化模拟"""

    def __init__(self, image_paths, labels, transform=None,
                 degrade_gamma=None, degrade_noise_sigma=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.degrade_gamma = degrade_gamma  # 亮度衰减 gamma 值
        self.degrade_noise_sigma = degrade_noise_sigma  # 高斯噪声标准差

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')

        # 退化模拟（亮度衰减）
        if self.degrade_gamma is not None:
            enhancer = ImageEnhance.Brightness(img)
            # gamma < 1 变暗，我们直接用因子乘，也可以用 gamma 校正
            # 这里简单用亮度因子：亮度因子 = gamma (0~1)
            img = enhancer.enhance(self.degrade_gamma)

        # 退化模拟（高斯噪声）
        if self.degrade_noise_sigma is not None:
            img_np = np.array(img).astype(np.float32)
            noise = np.random.normal(0, self.degrade_noise_sigma, img_np.shape)
            img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_np)

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        return img, label


def collect_data_paths(data_root, parking_list):
    """收集指定停车场下的所有图像路径及标签"""
    paths = []
    labels = []
    data_root = Path(data_root)
    label_map = {'Empty': 0, 'Occupied': 1}

    for parking in parking_list:
        parking_dir = data_root / parking
        if not parking_dir.exists():
            continue
        for weather in ['Cloudy', 'Sunny', 'Rainy']:
            weather_dir = parking_dir / weather
            if not weather_dir.exists():
                continue
            for status, label in label_map.items():
                status_dir = weather_dir / status
                if not status_dir.exists():
                    continue
                for img_file in status_dir.glob("*.jpg"):
                    paths.append(str(img_file))
                    labels.append(label)
    return paths, labels


def split_train_val_test(paths, labels, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """按比例划分数据集"""
    # 先分出 train + val 和 test
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        paths, labels, test_size=test_ratio, random_state=42, stratify=labels)
    # 再从 train_val 中分出 val
    relative_val_ratio = val_ratio / (train_ratio + val_ratio)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=relative_val_ratio,
        random_state=42, stratify=train_val_labels)
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)
