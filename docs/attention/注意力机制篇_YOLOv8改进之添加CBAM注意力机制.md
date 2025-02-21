![](https://i-blog.csdnimg.cn/blog_migrate/cc8c54c1a22de368e66081fd82d10c58.jpeg)

> 前言：Hello大家好，我是小哥谈。CBAM是一种用于图像分类的注意力机制，全称为Convolutional Block Attention Module。它可以自适应地学习每个通道和空间位置的重要性，从而提高模型的性能。CBAM由两个部分组成：通道注意力模块和空间注意力模块。通道注意力模块通过学习每个通道的重要性来增强特征图的表达能力，而空间注意力模块则通过学习每个空间位置的重要性来增强特征图的判别能力。本文所做出的改进即在YOLOv8的Neck网络中添加CBAM注意力机制！~🌈 

目录

[🚀1.基础概念][1.]

[🚀2.添加步骤][2.]

[🚀3.改进方法][3.]

[🍀🍀步骤1：block.py文件修改][1_block.py]

[🍀🍀步骤2：\_\_init\_\_.py文件修改][2_init_.py]

[🍀🍀步骤3：tasks.py文件修改][3_tasks.py]

[🍀🍀步骤4：创建自定义yaml文件][4_yaml]

[🍀🍀步骤5：新建train.py文件][5_train.py]

[🍀🍀步骤6：模型训练测试][6]

![](https://i-blog.csdnimg.cn/blog_migrate/9fccc2628d443c1d610693e75516c947.gif)

#### 🚀1.基础概念 

CBAM是一种用于图像分类的注意力机制，全称为Convolutional Block Attention Module。它可以自适应地学习每个通道和空间位置的重要性，从而提高模型的性能。CBAM由两个部分组成：通道注意力模块和空间注意力模块。通道注意力模块通过学习每个通道的重要性来增强特征图的表达能力，而空间注意力模块则通过学习每个空间位置的重要性来增强特征图的判别能力。

通道注意力模块首先通过全局平均池化操作来获取每个通道的全局信息，然后通过两个全连接层来学习每个通道的权重。最后，将这些权重应用于原始特征图上，以增强重要通道的响应。

空间注意力模块则通过一个类似于SENet中的Squeeze-and-Excitation结构来学习每个空间位置的权重。具体来说，它首先通过一个全局平均池化操作来获取每个通道的全局信息，然后通过两个全连接层来学习每个通道的权重。最后，将这些权重应用于原始特征图上，以增强重要位置的响应。

CBAM方法已经在多个图像分类任务中取得了优秀的表现，证明了其有效性。

![](https://i-blog.csdnimg.cn/blog_migrate/2802bae74b4f24561f5dba1e9dfe202b.png)

CBAM包括两个主要组成部分: 空间注意力模块和通道注意力模块。它们可以作为插件，轻松地集成到现有的CNN架构中，以提高性能。

空间注意力模块 (Spatial Attention Module) : 空间注意力模块的目的是为输入特征图的每个位置分配一个注意力权重。这些权重可帮助网络集中注意力于感兴趣的区域。

空间注意力模块主要包括以下步骤：👇

1.  对输入特征图进行全局平均池化和全局最大池化操作。
2.  将池化后的特征图按通道相加，得到两个1维向量。
3.  对这两个向量进行点积，形成一个注意力权重矩阵。
4.  将注意力权重矩阵应用于输入特征图，得到空间注意力调整后的特征图。

![](https://i-blog.csdnimg.cn/blog_migrate/944794ea83e1dd6f69ac3e4d9b28a0cd.png)

通道注意力模块 (ChannelAttention Module) : 通道注意力模块旨在为输入特征图的每个通道分配一个权重，从而强调重要的通道并抑制不太重要的通道。

通道注意力模块主要包括以下步骤:👇

1.  对输入特征图进行全局平均池化和全局最大池化操作。
2.  使用全连接层 (MLP) 学习每个通道的权重。
3.  将学到的权重应用于输入特征图，得到通道注意力调整后的特征图。

最后，将空间注意力模块和通道注意力模块的输出相加，形成CBAM调整后的特征图。这个特征图可以继续传递到CNN的下一层。实验证明，CBAM在各种计算机视觉任务上都能显著提高性能，如图像分类、目标检测和语义分割等。它是一种通用的注意力机制，可以与任何卷积神经网络架构相结合。✅

> 说明：♨️♨️♨️
> 
> CBAM的核心：
> 
> 应用了Channel Attention Module（通道注意模块）和Spatial Attention Module（空间注意模块）的结合，对输入进来的特征层分别进行通道注意力模块和空间注意力模块的处理。其中，通道注意力用于处理特征图通道的分配关系，而空间注意力可使神经网络更加关注图像中对分类起决定作用的像素区域而忽略无关紧要的区域。同时对这两个维度进行注意力的分配增强了注意力机制对模型性能的提升效果。

![](https://i-blog.csdnimg.cn/blog_migrate/51d23963f4fc3edb69cf4a4b7e11830c.png)

> 论文题目：《CBAM：Convolutional Block Attention Module》 
> 
> 论文地址： [https://arxiv.org/pdf/1807.06521.pdf][https_arxiv.org_pdf_1807.06521.pdf]
> 
> 代码实现： [GitHub - luuuyi/CBAM.PyTorch: Non-official implement of Paper：CBAM: Convolutional Block Attention Module][GitHub - luuuyi_CBAM.PyTorch_ Non-official implement of Paper_CBAM_ Convolutional Block Attention Module]

#### 🚀2.添加步骤 

针对本文的改进，具体步骤如下所示：👇

步骤1：block.py文件修改

步骤2：\_\_init\_\_.py文件修改

步骤3：tasks.py文件修改

步骤4：创建自定义yaml文件

步骤5：新建train.py文件

步骤6：模型训练测试

#### 🚀3.改进方法 

##### 🍀🍀步骤1：block.py文件修改 

在源码中找到block.py文件，具体位置是ultralytics/nn/modules/block.py，然后将CBAM模块代码添加到block.py文件末尾位置。 

CBAM模块代码：

```java
# CBAM模块代码
# By CSDN 小哥谈
class ChannelAttentionModule(nn.Module):
    def __init__(self, c1, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        mid_channel = c1 // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
 
        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=c1, out_features=mid_channel),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(in_features=mid_channel, out_features=c1)
        )
        self.act = nn.Sigmoid()
        #self.act=nn.SiLU()
    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
        return self.act(avgout + maxout)
        
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.act = nn.Sigmoid()
    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.act(self.conv2d(out))
        return out
 
class CBAM(nn.Module):
    def __init__(self, c1,c2):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(c1)
        self.spatial_attention = SpatialAttentionModule()
 
    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out
```

再然后，在block.py文件最上方下图所示位置加入CBAM。

![](https://i-blog.csdnimg.cn/blog_migrate/12f1be9aae75b0069a4cdcdeb2d4d551.png)

##### 🍀🍀步骤2：\_\_init\_\_.py文件修改 

在源码中找到\_\_init\_\_.py文件，具体位置是ultralytics/nn/modules/\_\_init\_\_.py。

修改1：加入CBAM，具体如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/fb845354a541441497c4f67ddfaee7b0.png)

修改2：加入CBAM，具体如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/1b001b713ece58b131bf009f5012feac.png)

##### 🍀🍀步骤3：tasks.py文件修改 

在源码中找到tasks.py文件，具体位置是ultralytics/nn/tasks.py。

然后找到parse\_model函数（736行左右），在下图中所示位置添加CBAM。

![](https://i-blog.csdnimg.cn/blog_migrate/f93a63b2b2ba9ea2c72e608b60daaaa2.png)

最后，在本文件中导入该模块，具体代码如下：

```java
from ultralytics.nn.modules.block import CBAM
```

##### 🍀🍀步骤4：创建自定义yaml文件 

在源码ultralytics/cfg/models/v8目录下创建yaml文件，并命名为：yolov8\_CBAM.yaml。具体如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/9bb0b61b05de53668749c0cfed1bd2e0.png)

yolov8\_CBAM.yaml文件完整代码如下所示：

```java
# Ultralytics YOLO 🚀, GPL-3.0 license
# YOLOv8 object detection model with P3-P5 outputs. For Usage examples see https://docs.ultralytics.com/tasks/detect

# Parameters
nc: 80  # number of classes
scales: # model compound scaling constants, i.e. 'model=yolov8n.yaml' will call yolov8.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.33, 0.25, 1024]  # YOLOv8n summary: 225 layers,  3157200 parameters,  3157184 gradients,   8.9 GFLOPs
  s: [0.33, 0.50, 1024]  # YOLOv8s summary: 225 layers, 11166560 parameters, 11166544 gradients,  28.8 GFLOPs
  m: [0.67, 0.75, 768]   # YOLOv8m summary: 295 layers, 25902640 parameters, 25902624 gradients,  79.3 GFLOPs
  l: [1.00, 1.00, 512]   # YOLOv8l summary: 365 layers, 43691520 parameters, 43691504 gradients, 165.7 GFLOPs
  x: [1.00, 1.25, 512]   # YOLOv8x summary: 365 layers, 68229648 parameters, 68229632 gradients, 258.5 GFLOPs

# YOLOv8.0n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]  # 9

# YOLOv8.0n head
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 6], 1, Concat, [1]]  # cat backbone P4
  - [-1, 3, C2f, [512]]  # 12
  - [-1, 1, CBAM, [512]]

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 4], 1, Concat, [1]]  # cat backbone P3
  - [-1, 3, C2f, [256]]  # 16 (P3/8-small)
  - [-1, 1, CBAM, [256]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]  # cat head P4
  - [-1, 3, C2f, [512]]  # 20 (P4/16-medium)
  - [-1, 1, CBAM, [512]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]]  # cat head P5
  - [-1, 3, C2f, [1024]]  # 24 (P5/32-large)
  - [-1, 1, CBAM, [1024]]

  - [[17, 21, 25], 1, Detect, [nc]]  # Detect(P3, P4, P5)
```

##### 🍀🍀步骤5：新建train.py文件 

在源码根目录下新建train.py文件，文件完整代码如下所示：

```java
from ultralytics import YOLO

# Load a model
model = YOLO(r'C:\Users\Lenovo\PycharmProjects\ultralytics-main\ultralytics\cfg\models\v8\yolov8_CBAM.yaml')  # build a new model from YAML
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
model = YOLO(r'C:\Users\Lenovo\PycharmProjects\ultralytics-main\ultralytics\cfg\models\v8\yolov8_CBAM.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
model.train(data=r'C:\Users\Lenovo\PycharmProjects\ultralytics-main\ultralytics\cfg\datasets\helmet.yaml', epochs=100,imgsz=640)
```

> 注意：一定要用绝对路径，以防发生报错。

##### 🍀🍀步骤6：模型训练测试 

在train.py文件，点击“运行”，在作者自制的安全帽佩戴检测数据集上，模型可以正常训练。

![](https://i-blog.csdnimg.cn/blog_migrate/9d1a846337717605208dc9447ea97539.png)

> 参数量对比：🌈
> 
> yolov8.yaml：225 layers, 3011238 parameters, 3011222 gradients, 8.2 GFLOPs
> 
> yolov8\_CBAM.yaml：273 layers, 3025046 parameters, 3025030 gradients, 8.2 GFLOPs

模型训练过程： 

![](https://i-blog.csdnimg.cn/blog_migrate/dab3aa9833fee5a0d7771adaf2dfc883.png)

模型训练结果：

![](https://i-blog.csdnimg.cn/blog_migrate/ef4a44f0cb474a483f9838bff9ddc27b.png)

![](https://i-blog.csdnimg.cn/blog_migrate/31e10fd23303a5f4ed2c4df1396aab20.gif)


[1.]: #%F0%9F%9A%801.%20%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5
[2.]: #%F0%9F%9A%802.%E6%B7%BB%E5%8A%A0%E6%AD%A5%E9%AA%A4
[3.]: #%F0%9F%9A%803.%E6%94%B9%E8%BF%9B%E6%96%B9%E6%B3%95
[1_block.py]: #%F0%9F%8D%80%F0%9F%8D%80%E6%AD%A5%E9%AA%A41%EF%BC%9Ablock.py%E6%96%87%E4%BB%B6%E4%BF%AE%E6%94%B9
[2_init_.py]: #%F0%9F%8D%80%F0%9F%8D%80%E6%AD%A5%E9%AA%A42%EF%BC%9A__init__.py%E6%96%87%E4%BB%B6%E4%BF%AE%E6%94%B9
[3_tasks.py]: #%F0%9F%8D%80%F0%9F%8D%80%E6%AD%A5%E9%AA%A43%EF%BC%9Atasks.py%E6%96%87%E4%BB%B6%E4%BF%AE%E6%94%B9
[4_yaml]: #%F0%9F%8D%80%F0%9F%8D%80%E6%AD%A5%E9%AA%A44%EF%BC%9A%E5%88%9B%E5%BB%BA%E8%87%AA%E5%AE%9A%E4%B9%89yaml%E6%96%87%E4%BB%B6
[5_train.py]: #%F0%9F%8D%80%F0%9F%8D%80%E6%AD%A5%E9%AA%A45%EF%BC%9A%E6%96%B0%E5%BB%BAtrain.py%E6%96%87%E4%BB%B6
[6]: #%F0%9F%8D%80%F0%9F%8D%80%E6%AD%A5%E9%AA%A46%EF%BC%9A%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%E6%B5%8B%E8%AF%95
[https_arxiv.org_pdf_1807.06521.pdf]: https://arxiv.org/pdf/1807.06521.pdf
[GitHub - luuuyi_CBAM.PyTorch_ Non-official implement of Paper_CBAM_ Convolutional Block Attention Module]: https://github.com/luuuyi/CBAM.PyTorch