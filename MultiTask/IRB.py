import torch.nn as nn


class InvertedResidualBlock(nn.Module):
    """https://arxiv.org/abs/1801.04381"""

    def __init__(self, in_channels, out_channels, expand_ratio, stride=1):
        super().__init__()
        self.stride = stride
        self.hidden_dim = in_channels * expand_ratio
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Boolean condition for Residual Block
        self.use_res_connect = self.stride == 1 and self.in_channels == self.out_channels
        self.output = self.separable_conv(in_channels=self.in_channels, hidden_dim=self.hidden_dim,
                                          out_channels=self.out_channels, stride=self.stride)

    # (Inverted) separable convolutions
    @staticmethod
    def separable_conv(in_channels, hidden_dim, out_channels, stride):
        return nn.Sequential(
            #  1x1 point-wise convolution
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
                nn.BatchNorm2d(hidden_dim, affine=True, eps=1e-5, momentum=0.1),
                nn.ReLU6(inplace=True)),
            # 3x3 depth-wise convolution
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim, affine=True, eps=1e-5, momentum=0.1),
                nn.ReLU6(inplace=True)),
            # 1x1 point-wise , no activation here
            nn.Sequential(
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
                nn.BatchNorm2d(out_channels, affine=True, eps=1e-5, momentum=0.1))
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.output(x)
        else:
            return self.output(x)
