import torchvision.models as models
import torch.nn as nn

def get_model():
    model = models.resnet18(pretrained=True)

    # modify first layer for grayscale
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # modify output
    model.fc = nn.Linear(model.fc.in_features, 1)

    return model
