"""
传感器通道封装：视觉通道、超声波通道。
"""

import sys
import time
from collections import deque
from typing import Union, Optional
import cv2
import numpy as np
from PIL import Image as PILImage

sys.path.append("..")
from ..CV_channel.CV_interface import ParkingOccupancyPredictor
from ..US_channel.US_interface import occupancy_probability


class VisualChannel:
    """视觉通道：内部估计光照，计算自适应权重"""
    def __init__(self, model_path: str, device: str = None,
                 lux0: float = 100.0, a1: float = 0.05,
                 fault_window: int = 3,
                 lux_scale: float = 2.0, lux_offset: float = 0.0):
        self.predictor = ParkingOccupancyPredictor(model_path, device)
        self.lux0 = lux0
        self.a1 = a1
        self.fault_window = fault_window
        self.lux_scale = lux_scale
        self.lux_offset = lux_offset

        self.prob_history = deque(maxlen=fault_window)
        self.timestamp_history = deque(maxlen=fault_window)
        self.is_fault = False
        self.weight = 0.99
        self.estimated_lux = None

    def estimate_lux(self, image: Union[np.ndarray, PILImage.Image]) -> float:
        if isinstance(image, PILImage.Image):
            img_np = np.array(image)
        else:
            img_np = image
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
        mean_brightness = np.mean(gray)
        estimated = self.lux_scale * mean_brightness + self.lux_offset
        return estimated

    def update_weight(self, image: Union[np.ndarray, PILImage.Image]) -> float:
        lux = self.estimate_lux(image)
        self.estimated_lux = lux
        z = self.a1 * (lux - self.lux0)
        self.weight = 1.0 / (1.0 + np.exp(-z))
        return self.weight

    def predict(self, image: Union[str, PILImage.Image, np.ndarray]) -> float:
        # 统一转为 PIL.Image
        if isinstance(image, str):
            img_pil = PILImage.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            img_pil = PILImage.fromarray(image.astype('uint8')).convert('RGB')
        elif isinstance(image, PILImage.Image):
            img_pil = image.convert('RGB')
        else:
            raise TypeError("不支持的图像类型")

        self.update_weight(img_pil)
        prob = self.predictor.predict(img_pil)

        current_time = time.time()
        self.prob_history.append(prob)
        self.timestamp_history.append(current_time)
        self._check_fault()
        return prob

    def _check_fault(self):
        if len(self.prob_history) < self.fault_window:
            self.is_fault = False
            return
        times = list(self.timestamp_history)
        if max(times) - min(times) > 10.0:
            self.is_fault = True
            return
        if np.var(self.prob_history) < 1e-6:
            self.is_fault = True
        else:
            self.is_fault = False

    def get_weight(self) -> float:
        return 0.0 if self.is_fault else self.weight


class UltrasonicChannel:
    """超声波通道，含温湿度补偿与权重计算"""
    def __init__(self, fault_window: int = 3, b1: float = 0.4, b2: float = 0.3):
        self.b1 = b1
        self.b2 = b2
        self.fault_window = fault_window
        self.distance_history = deque(maxlen=fault_window)
        self.prob_history = deque(maxlen=fault_window)
        self.timestamp_history = deque(maxlen=fault_window)
        self.is_fault = False

    def compute_weight(self, temp: float, humid: float) -> float:
        term1 = self.b1 * abs(temp - 20.0) / 40.0
        term2 = self.b2 * abs(humid - 50.0) / 50.0
        w = 1.0 - term1 - term2
        return max(0.0, float(w))

    def predict(self, distance: float, temp: float = 20.0, humid: float = 50.0) -> float:
        p_free = occupancy_probability(distance, temperature=temp, humidity=humid)
        p_occ = 1.0 - p_free
        current_time = time.time()
        self.distance_history.append(distance)
        self.prob_history.append(p_occ)
        self.timestamp_history.append(current_time)
        self._check_fault()
        return p_occ

    def _check_fault(self):
        if len(self.distance_history) < self.fault_window:
            self.is_fault = False
            return
        times = list(self.timestamp_history)
        if max(times) - min(times) > 10.0:
            self.is_fault = True
            return
        if np.var(self.distance_history) < 1e-6:
            self.is_fault = True
        else:
            self.is_fault = False

    def get_weight(self, temp: float, humid: float) -> float:
        if self.is_fault:
            return 0.0
        return self.compute_weight(temp, humid)


