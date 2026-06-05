"""
视觉预测器接口，使用 MobileNetV3-Small，输出占用概率。
修复了类型注解问题。
"""

from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image as PILImage
from torchvision import transforms, models


class ParkingOccupancyPredictor:
    def __init__(self, model_path: str, device: str = None):
        self.device = torch.device(device) if device else \
            torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.model = self._build_model()
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _build_model(self):
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 2)
        return model

    def predict(self, image: Union[str, PILImage.Image, np.ndarray]) -> float:
        if isinstance(image, str):
            img = PILImage.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            img = PILImage.fromarray(image.astype('uint8')).convert('RGB')
        elif isinstance(image, PILImage.Image):
            img = image.convert('RGB')
        else:
            raise TypeError("不支持的类型")

        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = F.softmax(outputs, dim=1)
            occupied_prob = probs[0, 1].item()
        return occupied_prob