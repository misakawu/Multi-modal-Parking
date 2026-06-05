# __init__.py
# - 停车位图像分类模型（使用MobileNetV3）
# - 数据集处理和加载
# - 图像预处理和增强
# - 模型评估和测试
# 主要组件:
# - ParkingOccupancyPredictor: 停车位占用状态预测器
# - ParkingDataset: 支持退化模拟的数据集类
# - 相关工具函数用于数据组织和评估
# 导出主要类和函数
from .CV_interface import ParkingOccupancyPredictor
from .parking_dataset import ParkingDataset

__all__ = [
    'ParkingOccupancyPredictor',
    'ParkingDataset'
]
