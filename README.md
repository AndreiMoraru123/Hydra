# Multi-task depth estimation & semantic segmentation

#### [Real-Time Joint Semantic Segmentation and Depth Estimation Using Asymmetric Annotations](https://arxiv.org/abs/1809.04766) implementation

![image](https://user-images.githubusercontent.com/81184255/193479326-8e8728d1-57cf-4f7f-8a06-1a7efe167b76.png)

<p align="center">
  <img src="https://thumbs.gfycat.com/DishonestCourteousArawana-size_restricted.gif" alt="KITTI" width="1200" height="150">
</p>

### Original Repo: [DrSleep](https://github.com/DrSleep/multi-task-refinenet)

## Building the [MobileNetV2 Encoder](https://arxiv.org/pdf/1801.04381.pdf)

![mobilenet](https://user-images.githubusercontent.com/81184255/221435565-077fd4fc-7d48-400d-8950-508003a3e589.png)

## Based on the Inverted Residual Block

<img align="left" src="https://user-images.githubusercontent.com/81184255/194058410-15522cc5-f41d-47dd-b471-081527d5b0e5.png" width = "390" height="640" />

```python
def separable_conv(in_channels, hidden_dim,
                   out_channels, stride):
     return nn.Sequential(
         #  1x1 point-wise convolution
         nn.Sequential(
             nn.Conv2d(in_channels, hidden_dim,
                       kernel_size=1, stride=1,
                       padding=0, groups=1,
                       bias=False),
             nn.BatchNorm2d(hidden_dim, affine=True,
                            eps=1e-5, momentum=0.1),
             nn.ReLU6(inplace=True)),
         # 3x3 depth-wise convolution, note "groups"
         nn.Sequential(
             nn.Conv2d(hidden_dim, hidden_dim,
                       kernel_size=3, stride=stride,
                       padding=1, groups=hidden_dim,
                       bias=False),
             nn.BatchNorm2d(hidden_dim, affine=True,
                            eps=1e-5, momentum=0.1),
             nn.ReLU6(inplace=True)),
         # 1x1 point-wise , no activation here
         nn.Sequential(
             nn.Conv2d(hidden_dim, out_channels,
                       kernel_size=1, stride=1,
                       padding=0, groups=1,
                       bias=False),
             nn.BatchNorm2d(out_channels,
                            affine=True, eps=1e-5,
                            momentum=0.1))
    )

```

```python
class InvertedResidualBlock(nn.Module):
    """https://arxiv.org/abs/1801.04381"""

    def __init__(self, in_channels, out_channels, expand_ratio, stride=1):
        super().__init__()
        self.stride = stride
        hidden_dim = in_channels * expand_ratio
        # Boolean condition for Residual Block
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        self.output = self.separable_conv(in_channels=in_channels, hidden_dim=hidden_dim, out_channels=out_channels,
                                          stride=self.stride)

    def forward(self, x):
        if self.use_res_connect:
	    # add residual
            return x + self.output(x)
        else:
            return self.output(x)
```

## Building the [Lightweight RefineNet Decoder](https://arxiv.org/pdf/1810.03272.pdf)

![image](https://user-images.githubusercontent.com/81184255/194060273-f525d0bc-5043-443d-ba74-baff3d2980dc.png)

## Based on Chained Residual Pooling 

![image](https://user-images.githubusercontent.com/81184255/194060637-acf1fed2-ab38-4edf-8767-107296912daf.png)

```python
class ChainedResidualPooling(nn.Module):
    def __init__(self, in_channels, out_channels, n_stages, groups=False):
        super().__init__()
        self.n_stages = n_stages
        for i in range(self.n_stages):
            setattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'),
                    nn.Conv2d(in_channels if (i == 0) else out_channels,
                              out_channels, kernel_size=1, stride=1,
                              bias=False, groups=in_channels if groups else 1))

        self.max_pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        
    def forward(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.max_pool(top)
            top = getattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'))(top)
            x = top + x
        return x
```

## Some more examples:

<img src="https://user-images.githubusercontent.com/81184255/193479360-faed9ca1-c54a-4b06-969b-8356a237fb56.gif" width="1000" height="150"/>

<img src="https://user-images.githubusercontent.com/81184255/193479381-82ad5f3e-3079-4381-a16b-7c4feea3ae25.gif" width="1000" height="150"/>

<img src="https://user-images.githubusercontent.com/81184255/193479395-499dccd1-b904-4205-b1b1-0dfaf81fd1f2.gif" width="1000" height="150"/>

#### The model was pretrained using the paper's KITTI weights 

#### As we can see, the model's predictions are lacking in lower visibility scenarios

<p align="center">
	<img src="https://user-images.githubusercontent.com/81184255/193479546-9218d405-7ade-45c7-bfbf-833ee16ebf4e.gif" width = "400" height="300" />
</p>

```bibtex
@misc{https://doi.org/10.48550/arxiv.1809.04766,
  doi = {10.48550/ARXIV.1809.04766},
  url = {https://arxiv.org/abs/1809.04766},
  author = {Nekrasov, Vladimir and Dharmasiri, Thanuja and Spek, Andrew and Drummond, Tom and Shen, Chunhua and Reid, Ian},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Real-Time Joint Semantic Segmentation and Depth Estimation Using Asymmetric Annotations},
  publisher = {arXiv},
  year = {2018},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

