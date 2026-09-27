"""Minimal RotFuse network used for tri-modal image fusion inference."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ConvLeakyRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)


class ConvBnLeakyRelu2d(nn.Module):
    # ``bn`` is retained for checkpoint compatibility with the released model.
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)


class ConvBnTanh2d(nn.Module):
    # ``bn`` is retained for checkpoint compatibility with the released model.
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return torch.tanh(self.conv(x)) / 2 + 0.5


class Conv1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Sobelxy(nn.Module):
    def __init__(self, channels):
        super().__init__()
        sobel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        self.convx = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.convy = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel))
        self.convy.weight.data.copy_(torch.from_numpy(sobel.T))

    def forward(self, x):
        return torch.abs(self.convx(x)) + torch.abs(self.convy(x))


class Channel_Attention(nn.Module):
    def __init__(self, channel, ratio=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.line1 = nn.Linear(channel, channel // ratio, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.line2 = nn.Linear(channel // ratio, channel, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        batch, channels, _, _ = x.shape
        weights = self.avgpool(x).view(batch, channels)
        weights = self.sig(self.line2(self.relu(self.line1(weights))))
        return x * weights.view(batch, channels, 1, 1)


class DenseBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvLeakyRelu2d(channels, channels)
        self.conv2 = ConvLeakyRelu2d(2 * channels, channels)

    def forward(self, x):
        x = torch.cat((x, self.conv1(x)), dim=1)
        return torch.cat((x, self.conv2(x)), dim=1)


class DenSoAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dense = DenseBlock(in_channels)
        self.convdown = Conv1(3 * in_channels, out_channels)
        self.sobelconv = Sobelxy(in_channels)
        self.convup = Conv1(in_channels, out_channels)
        self.cattention1 = Channel_Attention(out_channels)
        self.cattention2 = Channel_Attention(out_channels)

    def forward(self, x):
        dense = self.cattention1(self.convdown(self.dense(x)))
        edges = self.cattention2(self.convup(self.sobelconv(x)))
        return F.leaky_relu(dense + edges, negative_slope=0.1)


def Real_time_standard(x1, x2, x3):
    total = x1 + x2 + x3
    return (total - x1) / 2, (total - x2) / 2, (total - x3) / 2


class Self_Attention(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=1)
        self.sig = nn.Sigmoid()
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=1)
        self.conv3 = nn.Conv2d(channel, channel, kernel_size=1)

    def forward(self, x):
        query = self.sig(self.conv1(x)).transpose(3, 2)
        key = self.sig(self.conv2(x))
        attention = torch.matmul(query, key)
        return self.sig(self.conv3(torch.matmul(x, attention)))


class Rubik_Cube_Attention(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=7):
        super().__init__()
        self.channel_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.line1 = nn.Conv2d(channel, channel // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.line2 = nn.Conv2d(channel // ratio, channel, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.space = nn.Conv2d(1, 1, kernel_size, padding=3, bias=False)
        self.satt = Self_Attention(channel)
        self.conv = nn.Conv2d(channel, channel, 1)

    def forward(self, x):
        channel_weight = self.channel_avg_pool(x)
        channel_weight = self.sigmoid(self.relu(self.line2(self.relu(self.line1(channel_weight)))))
        spatial_weight = self.sigmoid(self.space(torch.mean(x, dim=1, keepdim=True)))
        attended = x * spatial_weight * channel_weight
        return self.sigmoid(self.conv(attended + 0.1 * self.satt(x)))


class Cross_Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention1 = Rubik_Cube_Attention(channels)
        self.attention2 = Rubik_Cube_Attention(channels)
        self.attention3 = Rubik_Cube_Attention(channels)

    def forward(self, x1, x2, x3):
        standard = Real_time_standard(x1, x2, x3)
        a1 = self.attention1(x1 - standard[0])
        a2 = self.attention2(x2 - standard[1])
        a3 = self.attention3(x3 - standard[2])
        return x1 + a1, x2 + a2, x3 + a3


class RotFuse(nn.Module):
    """RotFuse for visible, SWIR, and LWIR inputs, in that order."""

    def __init__(self, output=1):
        super().__init__()
        channels = [8, 16, 32]
        self.conv_vis = ConvLeakyRelu2d(1, channels[0])
        self.conv_inf = ConvLeakyRelu2d(1, channels[0])
        self.conv_tri = ConvLeakyRelu2d(1, channels[0])
        self.dsam1_vis = DenSoAM(channels[0], channels[1])
        self.dsam2_vis = DenSoAM(channels[1], channels[2])
        self.dsam1_inf = DenSoAM(channels[0], channels[1])
        self.dsam2_inf = DenSoAM(channels[1], channels[2])
        self.dsam1_tri = DenSoAM(channels[0], channels[1])
        self.dsam2_tri = DenSoAM(channels[1], channels[1])
        self.cro_att1 = Cross_Attention(channels[1])
        self.decode4 = ConvBnLeakyRelu2d(2 * channels[2] + channels[1], 3 * channels[1])
        self.decode3 = ConvBnLeakyRelu2d(3 * channels[1], 3 * channels[0])
        self.decode2 = ConvBnLeakyRelu2d(3 * channels[0], 2 * channels[0])
        self.decode_add = ConvBnLeakyRelu2d(2 * channels[0], channels[0])
        self.decode1 = ConvBnTanh2d(channels[0], output)

    def forward(self, visible, swir, lwir):
        vis0 = self.conv_vis(visible[:, :1])
        swir0 = self.conv_tri(swir[:, :1])
        lwir0 = self.conv_inf(lwir[:, :1])

        vis1 = self.dsam1_vis(vis0)
        swir1 = self.dsam1_tri(swir0)
        lwir1 = self.dsam1_inf(lwir0)
        vis1, swir1, lwir1 = self.cro_att1(vis1, swir1, lwir1)

        vis2 = self.dsam2_vis(vis1)
        swir2 = self.dsam2_tri(swir1)
        lwir2 = self.dsam2_inf(lwir1)
        fused = self.decode4(torch.cat((vis2, swir2, lwir2), dim=1))
        fused = self.decode3(fused)
        fused = self.decode2(fused)
        fused = self.decode_add(fused)
        return self.decode1(fused)


# Compatibility alias for the original checkpoint/code naming.
FusionNet_triple = RotFuse
