import torch.nn as nn


class ChainedResidualPooling(nn.Module):
    """https://arxiv.org/abs/1611.06612"""

    def __init__(self, in_channels, out_channels, n_stages, groups=False):
        super().__init__()
        self.n_stages = n_stages
        for i in range(self.n_stages):
            setattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'),
                    nn.Conv2d(in_channels if (i == 0) else out_channels,
                              out_channels, kernel_size=1, stride=1,
                              bias=False, groups=in_channels if groups else 1),
                    )

        self.max_pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.max_pool(top)
            top = getattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'))(top)
            x = top + x
        return x
