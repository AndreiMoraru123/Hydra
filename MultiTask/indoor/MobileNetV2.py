import torch.nn as nn
import sys
import os

sys.path.append(os.path.abspath('..'))
from IRB import InvertedResidualBlock


class MobileNetv2(nn.Module):
    def __init__(self):
        super().__init__()
        # expansion rate, output channels, number of repeats, stride
        CONFIG = [  # match all keys in the paper
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        # 32x32x3 input
        self.in_channels = 32
        self.num_layers = len(CONFIG)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, self.in_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels, affine=True, eps=1e-3, momentum=0.01),
            nn.ReLU6(inplace=True),
        )
        # 16x16x32
        self.layer2 = self._make_layer(CONFIG[0])
        # 16x16x16
        self.layer3 = self._make_layer(CONFIG[1])
        # 8x8x24
        self.layer4 = self._make_layer(CONFIG[2])
        # 4x4x32
        self.layer5 = self._make_layer(CONFIG[3])
        # 4x4x64
        self.layer6 = self._make_layer(CONFIG[4])
        # 4x4x96
        self.layer7 = self._make_layer(CONFIG[5])
        # 2x2x1607
        self.layer8 = self._make_layer(CONFIG[6])

        self.return_idx = [1, 2, 3, 4, 5, 6]
        self._out_c = [CONFIG[idx][1] for idx in self.return_idx]

    def _make_layer(self, config):
        # t, c, n, s
        (expand_ratio, out_channels, num_blocks, stride) = config
        layer = []
        for _ in range(num_blocks):
            layer.append(
                InvertedResidualBlock(in_channels=self.in_channels,
                                      out_channels=out_channels,
                                      stride=stride if _ == 0 else 1,
                                      expand_ratio=expand_ratio)
            )
            self.in_channels = out_channels
            # stride = 1
        return nn.Sequential(*layer)

    def forward(self, x):
        outs = []
        x = self.layer1(x)
        outs.append(self.layer2(x))  # 16, x / 2
        outs.append(self.layer3(outs[-1]))  # 24, x / 4
        outs.append(self.layer4(outs[-1]))  # 32, x / 8
        outs.append(self.layer5(outs[-1]))  # 64, x / 16
        outs.append(self.layer6(outs[-1]))  # 96, x / 16
        outs.append(self.layer7(outs[-1]))  # 160, x / 32
        outs.append(self.layer8(outs[-1]))  # 320, x / 32
        return [outs[idx] for idx in self.return_idx]