import torch
import torch.nn as nn
import torch.nn.functional as F


def global_median_pooling(x):
    median_pooled = torch.median(x.view(x.size(0), x.size(1), -1), dim=2)[0]
    return median_pooled.view(x.size(0), x.size(1), 1, 1)


class CoordinateAttention(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.conv = nn.Conv2d(inp, oup, 1, 1, 0)

    def forward(self, x):
        return self.conv(x)


class DSSA(nn.Module):
    def __init__(self, input_channels, reduction=4):
        super().__init__()
        self.internal_neurons = input_channels // reduction
        self.fc1 = nn.Conv2d(input_channels, self.internal_neurons, 1, bias=True)
        self.fc2 = nn.Conv2d(self.internal_neurons, input_channels, 1, bias=True)

    def forward(self, x):
        avg_pool = F.adaptive_avg_pool2d(x, (1, 1))
        max_pool = F.adaptive_max_pool2d(x, (1, 1))
        median_pool = global_median_pooling(x)

        def _mlp(pool):
            out = self.fc1(pool)
            out = F.relu(out, inplace=True)
            out = self.fc2(out)
            return out

        att = _mlp(avg_pool) + _mlp(max_pool) + _mlp(median_pool)
        return torch.sigmoid(att)


class MCSA(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.scale1 = nn.Conv2d(input_channels, input_channels, 1, padding=0, groups=input_channels)
        self.scale2 = nn.Conv2d(input_channels, input_channels, 3, padding=1, groups=input_channels)
        self.scale3 = nn.Conv2d(input_channels, input_channels, 5, padding=2, groups=input_channels)

        self.coord_att = CoordinateAttention(input_channels, input_channels, reduction=32)

        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_channels, 4, 1, bias=False),
            nn.Softmax(dim=1)
        )

        self.pointwise_conv = nn.Conv2d(input_channels, 1, 1, bias=False)

    def forward(self, x):
        s1 = self.scale1(x)
        s2 = self.scale2(x)
        s3 = self.scale3(x)
        s_ca = self.coord_att(x)

        weights = self.weight_generator(x)
        w1, w2, w3, w_ca = weights.chunk(4, dim=1)

        fused = w1 * s1 + w2 * s2 + w3 * s3 + w_ca * s_ca

        spatial_att = self.pointwise_conv(fused)
        return torch.sigmoid(spatial_att)


class HCA(nn.Module):
    def __init__(self, c1, c2, *args):
        super().__init__()

        self.spatial = MCSA(c1)
        self.channel = DSSA(c1)

        self.out = nn.Sequential(
            nn.Conv2d(c1, c2, 3, padding=1, bias=False),
            nn.GroupNorm(32, c2),
            nn.ReLU(inplace=True)
        )

        self.proj = nn.Identity()
        if c1 != c2:
            self.proj = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)

    def forward(self, x):
        identity = x
        channel_att_weights = self.channel(x)
        x_channel_att = x * channel_att_weights
        spatial_att_weights = self.spatial(x)
        x_spatial_att = x * spatial_att_weights
        fused_att = x_channel_att + x_spatial_att
        out = self.out(fused_att)

        return out + self.proj(identity)
