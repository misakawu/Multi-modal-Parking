from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from parking_dataset import ParkingDataset

# 图像预处理（统一尺寸并归一化）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 正常测试集加载
test_dataset = ParkingDataset(
    index_file=r"E:\DATASET\PKLot\index\puc_test.json",
    transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 弱光模拟测试（gamma=0.5）
lowlight_dataset = ParkingDataset(
    index_file=r"E:\DATASET\PKLot\index\puc_test.json",
    transform=transform,
    degrade_gamma=0.5
)

# 噪声模拟测试（sigma=20）
noise_dataset = ParkingDataset(
    index_file=r"E:\DATASET\PKLot\index\puc_test.json",
    transform=transform,
    degrade_noise_sigma=20
)

# CNR-EXT 测试集
cnr_dataset = ParkingDataset(
    index_file=r"E:\DATASET\CNR-EXT-Patches\index.json",
    transform=transform
)
