# tool
#
# 该包提供了多模态停车检测系统中的核心工具和传感器通道封装，主要包括：
# - 传感器通道封装类（视觉、超声波）
# - 数据仿真和模拟工具
# - 故障检测和权重计算机制
#
# 主要组件:
# - VisualChannel: 视觉通道封装，包含权重计算与故障检测
# - UltrasonicChannel: 超声波通道封装，含温湿度补偿、权重计算与故障检测
# - generate_samples: 超声波测距仿真样本生成器
# - humidity_noise_std: 湿度噪声标准差计算函数
#
# 功能特性:
# - 传感器故障检测机制
# - 环境参数补偿（温度、湿度、光照）
# - 动态权重调整
# - 数据仿真和测试支持

from .US_data_simulation import generate_samples, humidity_noise_std
# 导出主要类和函数
from .channels_packaging import VisualChannel, UltrasonicChannel

__all__ = [
    'VisualChannel',
    'UltrasonicChannel',
    'generate_samples',
    'humidity_noise_std'
]
