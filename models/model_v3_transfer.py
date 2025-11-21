import torch
import torch.nn as nn
from torchvision import models


class TransferModelV3(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, 2)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


if __name__ == "__main__":
    model = TransferModelV3(pretrained=False)
    x = torch.randn(4, 3, 224, 224)
    y = model(x)
    print("Output shape:", y.shape)  
