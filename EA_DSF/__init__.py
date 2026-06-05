# EA-DSF

# 该包实现了基于多模态传感器融合的智能停车位检测系统，主要包括：
# - 视觉通道（摄像头图像分析）
# - 超声波通道（距离测量）
# - D-S证据理论融合算法
# - 自适应加权融合方法
#
# 主要子模块:
# - CV_channel: 视觉通道，基于深度学习的图像分类
# - US_channel: 超声波通道，距离测量与温湿度补偿
# - tool: 工具模块，提供传感器封装和数据仿真
#
# 主要功能:
# - 多传感器数据采集与预处理
# - 环境参数补偿（温度、湿度、光照）
# - 传感器故障检测与容错
# - 智能融合决策与二次确认机制
# - 实时停车位占用状态检测

# 版本信息
__version__ = '1.0.0'
__author__ = 'EA-DSF Team'

# 导出主要组件
from .tool import (
    VisualChannel,
    UltrasonicChannel,
    generate_samples,
    humidity_noise_std
)

from .CV_channel import (
    ParkingOccupancyPredictor,
    ParkingDataset
)

__all__ = [
    # 版本信息
    '__version__',
    '__author__',

    # 传感器通道
    'VisualChannel',
    'UltrasonicChannel',

    # 视觉通道组件
    'ParkingOccupancyPredictor',
    'ParkingDataset',

]
