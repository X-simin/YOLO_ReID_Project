import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['osnet_x0_25', 'osnet_x0_5', 'osnet_x0_75', 'osnet_x1_0']

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class LightBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand=1):
        super(LightBlock, self).__init__()
        hidden = int(in_channels * expand)
        self.conv1 = Conv1x1(in_channels, hidden)
        self.conv2 = Conv3x3(hidden, hidden)
        self.conv3 = Conv1x1(hidden, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x += residual
        x = self.relu(x)
        return x

class OSNet(nn.Module):
    def __init__(self, num_classes, channels=64):
        super(OSNet, self).__init__()
        self.channels = channels
        self.num_classes = num_classes

        self.conv1 = ConvLayer(3, channels, 7, 2, 3)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.stage1 = nn.Sequential(
            LightBlock(channels, channels),
            LightBlock(channels, channels)
        )
        self.trans1 = Conv1x1(channels, channels*2)

        self.stage2 = nn.Sequential(
            LightBlock(channels*2, channels*2),
            LightBlock(channels*2, channels*2)
        )
        self.trans2 = Conv1x1(channels*2, channels*4)

        self.stage3 = nn.Sequential(
            LightBlock(channels*4, channels*4),
            LightBlock(channels*4, channels*4)
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels*4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.stage1(x)
        x = self.trans1(x)
        x = self.stage2(x)
        x = self.trans2(x)
        x = self.stage3(x)

        feat = self.global_pool(x)
        feat = torch.flatten(feat, 1)
        logits = self.fc(feat)
        return logits, feat

def osnet_x1_0(num_classes=751):
    return OSNet(num_classes, channels=64)