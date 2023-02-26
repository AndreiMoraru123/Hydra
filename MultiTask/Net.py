import torch.nn as nn
from IRB import InvertedResidualBlock
from CRP import ChainedResidualPooling


class HydraNet(nn.Module):
    def __init__(self):
        super().__init__()
        # two tasks, two heads
        self.num_tasks = 2
        # 6 segmentation classes, 1 depth
        self.num_classes = 6

        #######################
        # MobileNetV2 Encoder #
        #######################

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

        #####################
        # RefineNet Decoder #
        #####################

        self.conv8 = nn.Conv2d(320, 256, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv7 = nn.Conv2d(160, 256, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv6 = nn.Conv2d(96, 256, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv5 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv4 = nn.Conv2d(32, 256, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv3 = nn.Conv2d(24, 256, kernel_size=1, stride=1, padding=0, groups=1, bias=False)

        self.crp4 = self._make_crp(256, 256, 4, groups=False)
        self.crp3 = self._make_crp(256, 256, 4, groups=False)
        self.crp2 = self._make_crp(256, 256, 4, groups=False)
        self.crp1 = self._make_crp(256, 256, 4, groups=True)

        self.conv_adapt4 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_adapt3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_adapt2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False)

        self.pre_depth = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False, groups=256)
        self.depth = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1, bias=True, dilation=1)

        self.pre_segm = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False, groups=256)
        self.segm = nn.Conv2d(256, self.num_classes, kernel_size=3, stride=1, padding=1, bias=True, dilation=1)

        self.relu = nn.ReLU6(inplace=True)

        if self.num_tasks == 3:
            # normals
            self.pre_norm = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False, groups=256)
            self.norm = nn.Conv2d(256, 3, kernel_size=1, stride=1, padding=0, bias=True, dilation=1)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

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

    def mobilenet_encoder(self):
        pass

    @staticmethod
    def _make_crp(in_channels, out_channels, n_stages, groups=False):
        layers = [ChainedResidualPooling(in_channels, out_channels, n_stages, groups=groups)]
        return nn.Sequential(*layers)

    def refinenet_decoder(self):
        pass

    def forward(self, x):
        # MobileNet encoder
        x = self.layer1(x)
        x = self.layer2(x)
        l3 = self.layer3(x)
        l4 = self.layer4(l3)
        l5 = self.layer5(l4)
        l6 = self.layer6(l5)
        l7 = self.layer7(l6)
        l8 = self.layer8(l7)

        # RefineNet decoder
        l8 = self.conv8(l8)
        l7 = self.conv7(l7)
        l7 = self.relu(l8 + l7)
        l7 = self.crp4(l7)
        l7 = self.conv_adapt4(l7)
        # maximize output size for decoding
        l7 = nn.Upsample(size=l6.size()[2:], mode='bilinear', align_corners=False)(l7)

        l6 = self.conv6(l6)
        l5 = self.conv5(l5)
        l5 = self.relu(l6 + l5 + l7)
        l5 = self.crp3(l5)
        l5 = self.conv_adapt3(l5)
        # maximize output size for decoding
        l5 = nn.Upsample(size=l4.size()[2:], mode='bilinear', align_corners=False)(l5)

        l4 = self.conv4(l4)
        l4 = self.relu(l5 + l4)
        l4 = self.crp2(l4)
        l4 = self.conv_adapt2(l4)
        # maximize output size for decoding
        l4 = nn.Upsample(size=l3.size()[2:], mode='bilinear', align_corners=False)(l4)

        l3 = self.conv3(l3)
        l3 = self.relu(l3 + l4)
        l3 = self.crp1(l3)

        # heads
        depth = self.pre_depth(l3)
        depth = self.relu(depth)
        depth = self.depth(depth)

        segm = self.pre_segm(l3)
        segm = self.relu(segm)
        segm = self.segm(segm)

        if self.num_tasks == 3:  # normals
            norm = self.pre_norm(l3)
            norm = self.relu(norm)
            norm = self.norm(norm)
            return depth, segm, norm
        else:
            return depth, segm
