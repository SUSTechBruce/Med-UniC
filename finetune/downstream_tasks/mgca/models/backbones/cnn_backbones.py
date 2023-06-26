import torch.nn as nn
from torchvision import models as models_2d
# from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights


class Identity(nn.Module):
    """Identity layer to replace last fully connected layer"""

    def forward(self, x):
        return x


################################################################################
# ResNet Family
################################################################################


def resnet_18(pretrained=False):
    model = models_2d.resnet18(pretrained=False)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, 1024


def resnet_34(pretrained=False):
    model = models_2d.resnet34(pretrained=False)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, 1024


def resnet_50(pretrained=False):
    model = models_2d.resnet50(pretrained=False)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, 1024
