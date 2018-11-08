import torch

import network
import MobileNetV2
from constant import *
import torchvision
import great_networks
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model(pretrained=True, architecture="resnet34", is_train=True):

    if architecture == "resnet18":
        if pretrained:
            model = torchvision.models.resnet18(pretrained=pretrained)
            model.fc = torch.nn.Linear(512 * 1, NUM_CATEGORY)
        else:
            model = torchvision.models.resnet18(pretrained=pretrained, num_classes=NUM_CATEGORY)
        model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
    elif architecture == "resnet34":
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
    elif architecture == "mobilenetv2":
        model = MobileNetV2.MobileNetV2(n_class=NUM_CATEGORY, input_size=IMG_HEIGHT)
    elif architecture == "se_resnet50":
        if pretrained:
            model = great_networks.se_resnet50(pretrained=pretrained)
            model.last_linear = torch.nn.Linear(512 * 4, NUM_CATEGORY)
        else:
            model = great_networks.se_resnet50(num_classes=NUM_CATEGORY, pretrained=None)
        model.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
    elif architecture == "se_resnext50":
        if pretrained:
            model = great_networks.se_resnext50_32x4d(pretrained=pretrained)
            model.last_linear = torch.nn.Linear(512 * 4, NUM_CATEGORY)
        else:
            model = great_networks.se_resnext50_32x4d(num_classes=NUM_CATEGORY, pretrained=None)
        model.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
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


