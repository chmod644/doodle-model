import torch

import network
from constant import *
import torchvision

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model(pretrained=True, architecture="resnet34", is_train=True):

    if architecture == "resnet34":
        if pretrained:
            model = torchvision.models.resnet34(pretrained=pretrained)
            model.fc = torch.nn.Linear(512 * 1, NUM_CATEGORY)
        else:
            model = torchvision.models.resnet34(pretrained=pretrained, num_classes=NUM_CATEGORY)
        model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
    elif architecture == "resnet50":
        if pretrained:
            model = torchvision.models.resnet50(pretrained=pretrained)
            model.fc = torch.nn.Linear(512 * 4, NUM_CATEGORY)
        else:
            model = torchvision.models.resnet50(pretrained=pretrained, num_classes=NUM_CATEGORY)
        model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
    elif architecture == "original":
        model = network.ResNet(network.BasicBlock, [3, 4, 6, 3], num_classes=NUM_CATEGORY)
    else:
        raise ValueError()

    model.to(DEVICE)

    if is_train:
        model.train()
    else:
        model.eval()

    return model


