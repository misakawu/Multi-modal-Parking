import json

import numpy as np
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset


class ParkingDataset(Dataset):
    """停车位占用数据集，支持退化模拟（亮度衰减/高斯噪声）"""

    def __init__(self, index_file, transform=None,
                 degrade_gamma=None, degrade_noise_sigma=None):
        with open(index_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)  # 存储为普通 list，可序列化
        self.transform = transform  # torchvision transforms 可序列化
        self.degrade_gamma = degrade_gamma
        self.degrade_noise_sigma = degrade_noise_sigma

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = Image.open(item['path']).convert('RGB')
        label = item['label']

        # 亮度衰减
        if self.degrade_gamma is not None:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(self.degrade_gamma)

        # 高斯噪声
        if self.degrade_noise_sigma is not None:
            img_np = np.array(img).astype(np.float32)
            noise = np.random.normal(0, self.degrade_noise_sigma, img_np.shape)
            img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_np)

        if self.transform:
            img = self.transform(img)

        return img, label
