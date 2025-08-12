import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# --------------------------------------------------------------------
# 1×1 projection used in the encoder–decoder fusion
class AstroConv(nn.Sequential):
    def __init__(self, in_channels, out_channels=48):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


# --------------------------------------------------------------------
# Atrous Spatial Pyramid Pooling (ASPP) block
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv6  = self._conv_dil(in_channels, out_channels, 6)
        self.conv12 = self._conv_dil(in_channels, out_channels, 12)
        self.conv18 = self._conv_dil(in_channels, out_channels, 18)

        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    @staticmethod
    def _conv_dil(in_channels, out_channels, dilation):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[-2:]
        feats = [
            self.conv1x1(x),
            self.conv6(x),
            self.conv12(x),
            self.conv18(x),
            F.interpolate(self.image_pool(x), size=size,
                          mode='bilinear', align_corners=True),
        ]
        return self.project(torch.cat(feats, dim=1))


# --------------------------------------------------------------------
# ResNet‑50 broken into named stages for clarity
class ResNetBackbone(nn.Module):
    def __init__(self, stop_at: str):
        super().__init__()
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.stem   = nn.Sequential(m.conv1, m.bn1, m.relu)       # 64c
        self.pool   = m.maxpool                                    # 64c
        self.layer1 = m.layer1                                     # 256c
        self.layer2 = m.layer2                                     # 512c
        self.layer3 = m.layer3                                     # 1024c
        self.layer4 = m.layer4                                     # 2048c

        self.stop_at = stop_at

    def forward(self, x):
        x = self.stem(x)
        if self.stop_at == 'stem':  return x
        x = self.pool(x)
        if self.stop_at == 'pool':  return x
        x = self.layer1(x)
        if self.stop_at == 'layer1': return x
        x = self.layer2(x)
        if self.stop_at == 'layer2': return x
        x = self.layer3(x)
        if self.stop_at == 'layer3': return x
        x = self.layer4(x)
        return x


# --------------------------------------------------------------------
# Deeplab v3+ with ResNet‑50 backbone
class Deeplabv3Plus(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # backbone branches
        self.backbone_high = ResNetBackbone('layer3')   # 1024c, stride 16
        self.backbone_low  = ResNetBackbone('layer1')   # 256c,  stride 4

        # ASPP on the high‑level branch
        self.aspp = ASPP(1024, 256)

        # 1×1 projection on the low‑level branch (256→48)
        self.low_proj = AstroConv(256, 48)

        # decoder fusion
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # classifier
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        h, w = x.shape[-2:]

        x_high = self.backbone_high(x)          # stride 16
        x_low  = self.backbone_low(x)           # stride 4

        x_high = self.aspp(x_high)              # 256c, stride 16
        x_high = F.interpolate(
            x_high, size=x_low.shape[-2:], mode='bilinear', align_corners=True
        )

        x_low = self.low_proj(x_low)            # 48c,  stride 4

        x = torch.cat([x_low, x_high], dim=1)   # 304c, stride 4
        x = self.decoder(x)                     # 256c, stride 4
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        return self.classifier(x)


model = Deeplabv3Plus(num_classes=2)
x_test = torch.randn(1, 3, 512, 512)
model.eval()                     # ← this line
with torch.no_grad():
    y = model(x_test)
print(y.shape)
