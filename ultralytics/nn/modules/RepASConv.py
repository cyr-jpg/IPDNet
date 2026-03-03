import torch
import torch.nn as nn
import torch.nn.functional as F

class RepASConv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, deploy=False, use_act=True):
        super().__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.use_act = use_act
        self.activation = nn.SiLU() if use_act else nn.Identity()

        self.identity_valid = (in_channels == out_channels) and (stride == 1)

        if deploy:
            self.fused_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                groups=1,
                bias=True
            )
        else:
            self.branch1 = self._build_identity_branch()
            self.branch2 = self._build_branch2()
            self.branch3 = self._build_branch3()

    def _build_identity_branch(self):
        if self.identity_valid:
            return nn.Sequential(
                nn.Identity(),
                nn.BatchNorm2d(self.in_channels, eps=1e-5)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(
                    self.in_channels, self.out_channels,
                    kernel_size=1, stride=self.stride, padding=0,
                    groups=1, bias=False
                ),
                nn.BatchNorm2d(self.out_channels, eps=1e-5)
            )

    def _build_branch2(self):
        return nn.Sequential(
            nn.Conv2d(
                self.in_channels, self.in_channels,
                kernel_size=1, stride=self.stride, padding=0,
                groups=1, bias=False
            ),
            nn.BatchNorm2d(self.in_channels, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(
                self.in_channels, self.in_channels,
                kernel_size=(1, 3), stride=1, padding=(0, 1),
                groups=1, bias=False
            ),
            nn.BatchNorm2d(self.in_channels, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(
                self.in_channels, self.in_channels,
                kernel_size=(3, 1), stride=1, padding=(1, 0),
                groups=1, bias=False
            ),
            nn.BatchNorm2d(self.in_channels, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(
                self.in_channels, self.out_channels,
                kernel_size=1, stride=1, padding=0,
                groups=1, bias=False
            ),
            nn.BatchNorm2d(self.out_channels, eps=1e-5)
        )

    def _build_branch3(self):
        return nn.Sequential(
            nn.Conv2d(
                self.in_channels, self.in_channels,
                kernel_size=1, stride=self.stride, padding=0,
                groups=1, bias=False
            ),
            nn.BatchNorm2d(self.in_channels, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(
                self.in_channels, self.in_channels,
                kernel_size=(3, 1), stride=1, padding=(1, 0),
                groups=1, bias=False
            ),
            nn.BatchNorm2d(self.in_channels, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(
                self.in_channels, self.in_channels,
                kernel_size=(1, 3), stride=1, padding=(0, 1),
                groups=1, bias=False
            ),
            nn.BatchNorm2d(self.in_channels, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(
                self.in_channels, self.out_channels,
                kernel_size=1, stride=1, padding=0,
                groups=1, bias=False
            ),
            nn.BatchNorm2d(self.out_channels, eps=1e-5)
        )

    def _fuse_conv_bn(self, conv, bn):
        eps = bn.eps
        mu = bn.running_mean
        var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        std = torch.sqrt(var + eps)
        weight = conv.weight * (gamma / std).reshape(-1, 1, 1, 1)
        bias = beta - (mu * gamma) / std
        return weight, bias

    def _fuse_identity_branch(self):
        if self.identity_valid:
            bn = self.branch1[1]
            identity_kernel = torch.zeros(
                self.out_channels, self.in_channels, 1, 1, device=bn.weight.device
            )
            for i in range(min(self.in_channels, self.out_channels)):
                identity_kernel[i, i, 0, 0] = 1.0
            eps = bn.eps
            mu = bn.running_mean
            var = bn.running_var
            gamma = bn.weight
            beta = bn.bias
            std = torch.sqrt(var + eps)
            weight = identity_kernel * (gamma / std).reshape(-1, 1, 1, 1)
            bias = beta - (mu * gamma) / std
            return weight, bias
        else:
            conv = self.branch1[0]
            bn = self.branch1[1]
            return self._fuse_conv_bn(conv, bn)

    def _fuse_branch(self, branch):
        conv1, bn1 = branch[0], branch[1]
        w1, b1 = self._fuse_conv_bn(conv1, bn1)

        conv2, bn2 = branch[3], branch[4]
        w2, b2 = self._fuse_conv_bn(conv2, bn2)
        w12 = F.conv2d(
            w1.permute(1, 0, 2, 3),
            w2,
            padding=1,
            groups=1
        ).permute(1, 0, 2, 3)
        b12 = (w2 * b1.reshape(1, -1, 1, 1)).sum(dim=(1, 2, 3)) + b2

        conv3, bn3 = branch[6], branch[7]
        w3, b3 = self._fuse_conv_bn(conv3, bn3)
        w123 = F.conv2d(
            w12.permute(1, 0, 2, 3),
            w3,
            padding=1,
            groups=1
        ).permute(1, 0, 2, 3)
        b123 = (w3 * b12.reshape(1, -1, 1, 1)).sum(dim=(1, 2, 3)) + b3

        conv4, bn4 = branch[9], branch[10]
        w4, b4 = self._fuse_conv_bn(conv4, bn4)
        w_fused = F.conv2d(
            w123.permute(1, 0, 2, 3),
            w4,
            padding=0,
            groups=1
        ).permute(1, 0, 2, 3)
        b_fused = (w4 * b123.reshape(1, -1, 1, 1)).sum(dim=(1, 2, 3)) + b4

        return w_fused, b_fused

    def switch_to_deploy(self):
        if self.deploy:
            return
        self.deploy = True
        w_id, b_id = self._fuse_identity_branch()
        w2, b2 = self._fuse_branch(self.branch2)
        w3, b3 = self._fuse_branch(self.branch3)
        fused_weight = w_id + w2 + w3
        fused_bias = b_id + b2 + b3
        self.fused_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=self.stride,
            padding=0,
            groups=1,
            bias=True
        )
        self.fused_conv.weight.data = fused_weight
        self.fused_conv.bias.data = fused_bias
        del self.branch1, self.branch2, self.branch3

    def forward(self, x):
        if self.deploy:
            out = self.fused_conv(x)
            return self.activation(out)
        else:
            b1 = self.branch1(x)
            b2 = self.branch2(x)
            b3 = self.branch3(x)
            out = b1 + b2 + b3
            return self.activation(out)

class RepASConv5x5(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, deploy=False, use_act=True):
        super().__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.use_act = use_act
        self.activation = nn.SiLU() if use_act else nn.Identity()

        self.identity_valid = (in_channels == out_channels) and (stride == 1)

        if deploy:
            self.fused_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1,
                stride=stride, padding=0, bias=True
            )
        else:
            self.branch1 = self._build_identity_branch()
            self.branch2 = self._build_asym_branch(
                ks=(1, 5), ps=(0, 2),
                second_ks=(5, 1), second_ps=(2, 0)
            )
            self.branch3 = self._build_asym_branch(
                ks=(5, 1), ps=(2, 0),
                second_ks=(1, 5), second_ps=(0, 2)
            )

    def _build_identity_branch(self):
        if self.identity_valid:
            return nn.Sequential(
                nn.Identity(),
                nn.BatchNorm2d(self.in_channels, eps=1e-5)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, 1, self.stride, 0, bias=False),
                nn.BatchNorm2d(self.out_channels, eps=1e-5)
            )

    def _build_asym_branch(self, ks, ps, second_ks, second_ps):
        return nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 1, self.stride, 0, bias=False),
            nn.BatchNorm2d(self.in_channels, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(self.in_channels, self.in_channels, ks, 1, ps, bias=False),
            nn.BatchNorm2d(self.in_channels, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(self.in_channels, self.in_channels, second_ks, 1, second_ps, bias=False),
            nn.BatchNorm2d(self.in_channels, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.out_channels, eps=1e-5)
        )

    def _fuse_conv_bn(self, conv, bn):
        std = torch.sqrt(bn.running_var + bn.eps)
        weight = conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1)
        bias = bn.bias - (bn.running_mean * bn.weight) / std
        return weight, bias

    def _fuse_identity_branch(self):
        if self.identity_valid:
            bn = self.branch1[1]
            identity_kernel = torch.zeros(
                self.out_channels, self.in_channels, 1, 1,
                device=bn.weight.device
            )
            for i in range(min(self.in_channels, self.out_channels)):
                identity_kernel[i, i, 0, 0] = 1.0
            std = torch.sqrt(bn.running_var + bn.eps)
            weight = identity_kernel * (bn.weight / std).reshape(-1, 1, 1, 1)
            bias = bn.bias - (bn.running_mean * bn.weight) / std
            return weight, bias
        else:
            return self._fuse_conv_bn(self.branch1[0], self.branch1[1])

    def _fuse_branch(self, branch):
        w1, b1 = self._fuse_conv_bn(branch[0], branch[1])
        w2, b2 = self._fuse_conv_bn(branch[3], branch[4])
        w3, b3 = self._fuse_conv_bn(branch[6], branch[7])
        w4, b4 = self._fuse_conv_bn(branch[9], branch[10])

        w12 = F.conv2d(
            w1.permute(1, 0, 2, 3),
            w2,
            padding=(w2.shape[2] // 2, w2.shape[3] // 2)
        ).permute(1, 0, 2, 3)
        b12 = (w2 * b1.reshape(1, -1, 1, 1)).sum(dim=(1, 2, 3)) + b2

        w123 = F.conv2d(
            w12.permute(1, 0, 2, 3),
            w3,
            padding=(w3.shape[2] // 2, w3.shape[3] // 2)
        ).permute(1, 0, 2, 3)
        b123 = (w3 * b12.reshape(1, -1, 1, 1)).sum(dim=(1, 2, 3)) + b3

        w_fused = F.conv2d(
            w123.permute(1, 0, 2, 3),
            w4,
            padding=0
        ).permute(1, 0, 2, 3)
        b_fused = (w4 * b123.reshape(1, -1, 1, 1)).sum(dim=(1, 2, 3)) + b4

        if w_fused.shape[2:] != (1, 1):
            h, w = w_fused.shape[2:]
            w_fused = w_fused[:, :, h // 2:h // 2 + 1, w // 2:w // 2 + 1]

        return w_fused, b_fused

    def switch_to_deploy(self):
        if self.deploy:
            return
        self.deploy = True

        w_id, b_id = self._fuse_identity_branch()
        w2, b2 = self._fuse_branch(self.branch2)
        w3, b3 = self._fuse_branch(self.branch3)

        self.fused_conv = nn.Conv2d(
            self.in_channels, self.out_channels,
            1, self.stride, 0, bias=True
        )
        self.fused_conv.weight.data = w_id + w2 + w3
        self.fused_conv.bias.data = b_id + b2 + b3

        del self.branch1, self.branch2, self.branch3

    def forward(self, x):
        if self.deploy:
            return self.activation(self.fused_conv(x))
        return self.activation(
            self.branch1(x) + self.branch2(x) + self.branch3(x)
        )
