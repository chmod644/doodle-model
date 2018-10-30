import torch

import config
import torchvision


def create_resnet(pretrained=True, resnet_type="resnet34"):
    if resnet_type == "resnet34":
        model = torchvision.models.resnet34(pretrained=pretrained, num_classes=config.NUM_CLASSES)
    elif resnet_type == "resnet50":
        model = torchvision.models.resnet50(pretrained=pretrained, num_classes=config.NUM_CLASSES)
    else:
        raise ValueError()

    model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
    return model