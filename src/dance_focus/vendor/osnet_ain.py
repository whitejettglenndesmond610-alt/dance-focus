"""Minimal OSNet-AIN model definition from Torchreid.

Adapted from Kaiyang Zhou's deep-person-reid project at commit
f8cd150fdf77e8d9e1ed143b7f308c2c609ded50. Distributed under the MIT License.
"""

from __future__ import annotations

from torch import nn
from torch.nn import functional as F


class ConvLayer(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, IN=False
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = (
            nn.InstanceNorm2d(out_channels, affine=True)
            if IN
            else nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None

    def forward(self, x):
        x = self.conv(x)
        return self.bn(x) if self.bn is not None else x


class LightConv3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            padding=1,
            bias=False,
            groups=out_channels,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class LightConvStream(nn.Module):
    def __init__(self, in_channels, out_channels, depth):
        super().__init__()
        layers = [LightConv3x3(in_channels, out_channels)]
        layers.extend(
            LightConv3x3(out_channels, out_channels) for _ in range(depth - 1)
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ChannelGate(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.norm1 = None
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x):
        gates = self.global_avgpool(x)
        gates = self.relu(self.fc1(gates))
        gates = self.gate_activation(self.fc2(gates))
        return x * gates


class OSBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels // 4
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2 = nn.ModuleList(
            LightConvStream(mid_channels, mid_channels, depth)
            for depth in range(1, 5)
        )
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = (
            Conv1x1Linear(in_channels, out_channels)
            if in_channels != out_channels
            else None
        )

    def forward(self, x):
        identity = x
        features = self.conv1(x)
        combined = sum(self.gate(stream(features)) for stream in self.conv2)
        output = self.conv3(combined)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return F.relu(output + identity)


class OSBlockINin(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels // 4
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2 = nn.ModuleList(
            LightConvStream(mid_channels, mid_channels, depth)
            for depth in range(1, 5)
        )
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels, bn=False)
        self.downsample = (
            Conv1x1Linear(in_channels, out_channels)
            if in_channels != out_channels
            else None
        )
        self.IN = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        identity = x
        features = self.conv1(x)
        combined = sum(self.gate(stream(features)) for stream in self.conv2)
        output = self.IN(self.conv3(combined))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return F.relu(output + identity)


class OSNet(nn.Module):
    def __init__(self, num_classes=1, feature_dim=512):
        super().__init__()
        self.loss = "softmax"
        self.feature_dim = feature_dim
        self.conv1 = ConvLayer(3, 64, 7, stride=2, padding=3, IN=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer([OSBlockINin, OSBlockINin], 64, 256)
        self.pool2 = nn.Sequential(Conv1x1(256, 256), nn.AvgPool2d(2, stride=2))
        self.conv3 = self._make_layer([OSBlock, OSBlockINin], 256, 384)
        self.pool3 = nn.Sequential(Conv1x1(384, 384), nn.AvgPool2d(2, stride=2))
        self.conv4 = self._make_layer([OSBlockINin, OSBlock], 384, 512)
        self.conv5 = Conv1x1(512, 512)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)
        self._init_params()

    @staticmethod
    def _make_layer(blocks, in_channels, out_channels):
        return nn.Sequential(
            blocks[0](in_channels, out_channels),
            blocks[1](out_channels, out_channels),
        )

    def _init_params(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.InstanceNorm2d)):
                if layer.weight is not None:
                    nn.init.constant_(layer.weight, 1)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.01)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.conv5(self.conv4(x))
        features = self.global_avgpool(x).flatten(1)
        features = self.fc(features)
        return self.classifier(features) if self.training else features


def osnet_ain_x1_0(num_classes=1):
    return OSNet(num_classes=num_classes)
