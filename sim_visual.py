# sim_visual.py
"""
视觉通道退化注入。
复用 PKLot 数据集，在加载图像时动态施加退化（弱光、噪声、油污、水雾、眩光）。
不依赖占用序列 s，直接通过 ParkingDataset 读取真实车位图像。
"""
import cv2
from PIL import Image
from torch.utils.data import Dataset

from CV_channel_backend.parking_dataset import ParkingDataset
from config import *

rng_visual = np.random.RandomState(SEED + 3)


# ================= 退化函数 =================
def degrade_lowlight(img, gamma):
    """伽马校正 + 高斯噪声（gamma<1时使用倒数实现变暗）"""
    x = img.astype(np.float32) / 255.0
    # 使用 1/gamma 确保图像变暗：gamma=0.5 → 实际指数=2.0
    x_gamma = np.power(np.clip(x, 1e-5, 1), 1.0 / gamma)
    y = x_gamma * 255.0
    return np.clip(y, 0, 255).astype(np.uint8)


def degrade_oil_dirt(img, ratio):
    """随机半透明椭圆模拟油污/灰尘"""
    H, W = img.shape[:2]
    M = np.zeros((H, W), dtype=np.float32)
    target_area = ratio * H * W
    area = 0.0
    while area < target_area:
        cx, cy = rng_visual.randint(0, W), rng_visual.randint(0, H)
        a = rng_visual.uniform(5, 30)
        b = rng_visual.uniform(5, 30)
        angle = rng_visual.uniform(0, np.pi)
        alpha = rng_visual.uniform(0.4, 0.8)
        y, x = np.ogrid[:H, :W]
        x_rot = (x - cx) * np.cos(angle) + (y - cy) * np.sin(angle)
        y_rot = -(x - cx) * np.sin(angle) + (y - cy) * np.cos(angle)
        mask_ellipse = ((x_rot / a) ** 2 + (y_rot / b) ** 2) <= 1
        # 计算新增的实际覆盖面积（去重）
        new_pixels = np.sum(mask_ellipse & (M == 0))
        M[mask_ellipse] = np.maximum(M[mask_ellipse], alpha)
        area += new_pixels  # 累加实际新增像素数
    I_blur = cv2.GaussianBlur(img, (3, 3), 3)
    I_dirty = np.clip(I_blur * np.array([0.6, 0.5, 0.4]), 0, 255).astype(np.uint8)
    M_3c = np.stack([M, M, M], axis=2)
    return np.clip((1 - M_3c) * img + M_3c * I_dirty, 0, 255).astype(np.uint8)


def degrade_water_mist(img, beta):
    """大气散射模型模拟水雾"""
    H, W = img.shape[:2]
    dark = cv2.erode(img, np.ones((15, 15), np.uint8)).min(axis=2)
    top_pixels = np.percentile(dark, 99.9)
    A = np.mean(img[dark >= top_pixels], axis=0)
    if A is None or np.isnan(A).any():
        A = np.array([200, 200, 200])
    cx, cy = W // 2, H // 2
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    d_map = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / np.sqrt(cx ** 2 + cy ** 2)
    t = np.exp(-beta * d_map)
    t_3c = np.stack([t, t, t], axis=2)
    I_fog = np.clip(t_3c * img + (1 - t_3c) * A, 0, 255).astype(np.uint8)
    # 随机水滴
    n_drops = rng_visual.randint(20, 100)
    for _ in range(n_drops):
        cx_d = rng_visual.randint(0, W)
        cy_d = rng_visual.randint(0, H)
        a = rng_visual.randint(5, 15)
        b = rng_visual.randint(5, 15)
        angle = rng_visual.uniform(0, np.pi)
        y, x = np.ogrid[:H, :W]
        xr = (x - cx_d) * np.cos(angle) + (y - cy_d) * np.sin(angle)
        yr = -(x - cx_d) * np.sin(angle) + (y - cy_d) * np.cos(angle)
        mask = ((xr / a) ** 2 + (yr / b) ** 2) <= 1
        I_fog[mask] = cv2.GaussianBlur(I_fog, (5, 5), 0)[mask]
    return I_fog


def degrade_glare(img):
    """局部眩光（暖色高斯亮斑）"""
    H, W = img.shape[:2]
    cx = rng_visual.randint(0, W)
    cy = rng_visual.randint(H // 2, H)
    r = rng_visual.uniform(*GLARE_RADIUS_RANGE)
    delta = rng_visual.uniform(*GLARE_INTENSITY_RANGE)
    y, x = np.ogrid[:H, :W]
    G = delta * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * r ** 2))
    I_out = img.astype(np.float32) + np.stack([G, G, G * 0.95], axis=2)
    return np.clip(I_out, 0, 255).astype(np.uint8)


# ================= 视觉测试数据集 =================
class VisualTestDataset(Dataset):
    """
    加载 PKLot PUC 测试集，并动态注入一种退化。
    degrade_type: 'gamma', 'noise', 'oil_dirt', 'water_mist', 'glare' 或 None
    degrade_param: 对应的退化程度数值
    """

    def __init__(self, index_file, transform=None, degrade_type=None, degrade_param=None):
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"索引文件不存在: {index_file}")
        self.base = ParkingDataset(index_file)
        self.transform = transform
        self.degrade_type = degrade_type
        self.degrade_param = degrade_param

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img_pil, label = self.base[idx]
        if self.degrade_type is None:
            return img_pil, label

        img_np = np.array(img_pil)
        if self.degrade_type == 'gamma':
            img_np = degrade_lowlight(img_np, self.degrade_param, 0)
        elif self.degrade_type == 'noise':
            img_np = degrade_lowlight(img_np, 1.0, self.degrade_param)
        elif self.degrade_type == 'oil_dirt':
            img_np = degrade_oil_dirt(img_np, self.degrade_param)
        elif self.degrade_type == 'water_mist':
            img_np = degrade_water_mist(img_np, self.degrade_param)
        elif self.degrade_type == 'glare':
            img_np = degrade_glare(img_np)
        else:
            raise ValueError(f"不支持的退化类型: {self.degrade_type}")
        img = Image.fromarray(img_np)
        if self.transform is not None:
            img_np = self.transform(img)
        return img_np, label
