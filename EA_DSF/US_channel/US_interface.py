"""
US_interface.py
超声波车位占用概率计算接口（软判决）
支持温湿度补偿，输出 Sigmoid 概率 P_u
"""

import numpy as np


def sound_speed(T, H=0.0):
    """
    计算声速 (m/s)，考虑温度与湿度影响
    c = 331.3 + 0.606·T + 1.27e-3·H
    """
    return 331.3 + 0.606 * T + 0.00127 * H


def occupancy_probability(distance, temperature=20.0, humidity=50.0,
                          d_threshold=1.8, k=20.0,
                          T_ref=20.0, H_ref=50.0):
    """
    计算车位空闲概率 P_u

    Parameters
    ----------
    distance : float or array_like
        传感器测距值 (m)，假设已转换为距离
    temperature : float
        环境温度 (°C)
    humidity : float
        环境相对湿度 (%)
    d_threshold : float
        空位/占用判定阈值 (m)，默认为 1.8
    k : float
        Sigmoid 斜率系数，决定判决的软硬程度,温度系数
    T_ref, H_ref : float
        传感器出厂标定时的参考温湿度

    Returns
    -------
    P_u : float or ndarray
        空闲概率，∈ [0,1]；接近 1 为空位，接近 0 为占用
    """
    d_meas = np.asarray(distance)

    c_ref = sound_speed(T_ref, H_ref)
    c_curr = sound_speed(temperature, humidity)
    d_corrected = d_meas * (c_curr / c_ref)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    x = k * (d_threshold - d_corrected)
    P_u = sigmoid(x)
    if np.isscalar(distance):
        return float(P_u)
    return P_u
