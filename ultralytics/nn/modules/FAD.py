import torch
import torch.nn as nn

def haar_wavelet_downsample(x, alpha=1.2, beta=1.4):
    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]

    LL = (x00 + x01 + x10 + x11) * 0.5
    LH = (x00 + x11 - x01 - x10) * 0.5 * alpha
    HL = (x00 + x01 - x10 - x11) * 0.5 * alpha
    HH = (x00 + x10 - x01 - x11) * 0.5 * beta

    return torch.cat([LL, LH, HL, HH], dim=1)


class FAD(nn.Module):
    def __init__(self, in_channels, out_channels,
                 alpha_h=1.2, beta_h=1.4,
                 activation=nn.GELU):
        super().__init__()

        self.alpha_h = alpha_h
        self.beta_h = beta_h

        self.spatial_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.wavelet_process = nn.Sequential(
            nn.Conv2d(in_channels * 4, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.fusion_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1, bias=False),
            activation(),
            nn.Conv2d(out_channels // 4, out_channels, 1, bias=False),
            nn.Sigmoid()
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 2, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.final_activation = activation()

    def forward(self, x):

        spatial_out = self.spatial_path(x)

        dwt_feat = haar_wavelet_downsample(
            x,
            alpha=self.alpha_h,
            beta=self.beta_h
        )
      
        freq_out = self.wavelet_process(dwt_feat)
        shortcut_out = self.shortcut(x)
        fused_base = spatial_out + freq_out
        alpha = self.fusion_gate(fused_base)
        weighted_fused = fused_base * alpha
        out = weighted_fused + shortcut_out
        return self.final_activation(out)


if __name__ == '__main__':
    x = torch.randn(2, 64, 64, 64)
    model = FAD(64, 128)
    y = model(x)
    print(x.shape)
    print(y.shape)
