# Hydra
## Multi task depth and segmentation estimation for computer vision in self driving cars

My version of the <em> Real-Time Joint Semantic Segmentation and Depth Estimation Using Asymmetric Annotations </em> paper


![image](https://user-images.githubusercontent.com/81184255/193479326-8e8728d1-57cf-4f7f-8a06-1a7efe167b76.png)

#### Results:

##### Sunny:

![sunny](https://user-images.githubusercontent.com/81184255/193479360-faed9ca1-c54a-4b06-969b-8356a237fb56.gif)

#### Rainy:

![rainy](https://user-images.githubusercontent.com/81184255/193479381-82ad5f3e-3079-4381-a16b-7c4feea3ae25.gif)

#### Nightfall:

![night](https://user-images.githubusercontent.com/81184255/193479395-499dccd1-b904-4205-b1b1-0dfaf81fd1f2.gif)

##### The model was pretrained using the KITTI dataset weights 

As you can see, the model's predictions are lacking in lower visibility scenarios

# Building the MobileNetV2 encoder: The Inverted Residual Block

![image](https://user-images.githubusercontent.com/81184255/194058410-15522cc5-f41d-47dd-b471-081527d5b0e5.png)


```
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
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim,
                          bias=False),
                nn.BatchNorm2d(hidden_dim, affine=True, eps=1e-5, momentum=0.1),
                nn.ReLU6(inplace=True)),
            # 1x1 point-wise , no activation here
            nn.Sequential(
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
                nn.BatchNorm2d(out_channels, affine=True, eps=1e-5, momentum=0.1))
        )
```


```
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

![hail_hydra](https://user-images.githubusercontent.com/81184255/193479546-9218d405-7ade-45c7-bfbf-833ee16ebf4e.gif)
