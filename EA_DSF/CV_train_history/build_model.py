import torch.nn as nn
import torchvision.models as models


def build_mobilenetv3_small(num_classes=2, freeze_layers=True):
    """
    构建 MobileNetV3-Small，替换分类头。
    freeze_layers=True: 仅训练最后的分类层和部分高层（例如最后的几个 block）
    """
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)

    if freeze_layers:
        # 冻结所有参数
        for param in model.parameters():
            param.requires_grad = False

        # 解冻最后的分类层
        for param in model.classifier.parameters():
            param.requires_grad = True

        # 可选：解冻最后一个卷积块（features 的最后几个层），以微调高层特征
        # MobileNetV3-Small 的 features 共 16 个模块（0-15），解冻最后两个
        for name, param in model.features.named_parameters():
            if name.startswith('14') or name.startswith('15'):
                param.requires_grad = True
    else:
        # 训练全部
        pass

    # 替换分类头
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model
