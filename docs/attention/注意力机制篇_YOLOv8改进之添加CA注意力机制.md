![](https://i-blog.csdnimg.cn/blog_migrate/001f08ad0b675d661de87bfb9b5e3076.jpeg)

> 前言：Hello大家好，我是小哥谈。CA（Coordinate Attention）注意力机制是一种用于计算机视觉任务的注意力机制，它可以通过学习通道之间的关系来提高模型的性能。本文所做出的改进即在YOLOv8主干网络中添加CA注意力机制！~🌈 

目录

[🚀1.基础概念][1.]

[🚀2.网络结构][2.]

[🚀3.添加步骤][3.]

[🚀4.改进方法][4.]

[🍀🍀步骤1：conv.py文件修改][1_conv.py]

[🍀🍀步骤2：\_\_init\_\_.py文件修改][2_init_.py]

[🍀🍀步骤3：tasks.py文件修改][3_tasks.py]

[🍀🍀步骤4：创建自定义yaml文件][4_yaml]

[🍀🍀步骤5：新建train.py文件][5_train.py]

[🍀🍀步骤6：模型训练测试][6]

![](https://i-blog.csdnimg.cn/blog_migrate/5bb99c110d340983d12842c360b5f0c0.gif)

#### 🚀1. 基础概念 

CA（Coordinate Attention）注意力机制是一种用于计算机视觉任务的注意力机制，它可以通过学习通道之间的关系来提高模型的性能。CA注意力机制的基本思想是，对于给定的输入特征图，通过学习通道之间的关系来计算每个通道的权重，然后将这些权重应用于输入特征图中的每个像素点，以产生加权特征图。

具体来说，CA注意力机制包括两个步骤：通道特征提取和通道注意力计算。在通道特征提取阶段，我们使用一个全局平均池化层来计算每个通道的平均值和最大值，然后将它们连接起来并通过一个全连接层来产生通道特征。在通道注意力计算阶段，我们使用一个sigmoid函数来将通道特征映射到\[0,1\]范围内，并将其应用于输入特征图中的每个像素点，以产生加权特征图。

其原理如下：👇

1.  输入特征经过卷积等操作得到中间特征表示。
2.  中间特征表示经过两个并行的操作：全局平均池化和全局最大池化，得到全局特征描述。
3.  全局特征描述通过两个全连接层生成注意力权重。
4.  注意力权重与中间特征表示相乘，得到加权后的特征表示。
5.  加权后的特征表示经过适当的调整（如残差连接）后，作为下一层的输入。

CA注意力的实现如下图所示，可以认为分为两个并行阶段：

![](https://i-blog.csdnimg.cn/blog_migrate/1fd045e00a8a5ce67dba7c63696bfd38.png)

将输入特征图分别在为宽度和高度两个方向分别进行全局平均池化，分别获得在宽度和高度两个方向的特征图。假设输入进来的特征层的形状为\[C, H, W\]，在经过宽方向的平均池化后，获得的特征层shape为\[C, H, 1\]，此时我们将特征映射到了高维度上；在经过高方向的平均池化后，获得的特征层shape为\[C, 1, W\]，此时我们将特征映射到了宽维度上。

然后将两个并行阶段合并，将宽和高转置到同一个维度，然后进行堆叠，将宽高特征合并在一起，此时我们获得的特征层为：\[C, 1, H+W\]，利用卷积+标准化+激活函数获得特征。

之后再次分开为两个并行阶段，再将宽高分开成为：\[C, 1, H\]和\[C, 1, W\]，之后进行转置。获得两个特征层\[C, H, 1\]和\[C, 1, W\]。

然后利用1x1卷积调整通道数后取sigmoid获得宽高维度上的注意力情况，乘上原有的特征就是CA注意力机制。

加入CA注意力机制的好处包括：

1.  增强特征表达：CA注意力机制能够自适应地选择和调整不同通道的特征权重，从而更好地表达输入数据。它可以帮助模型发现和利用输入数据中重要的通道信息，提高特征的判别能力和区分性。
2.  减少冗余信息：通过抑制不重要的通道，CA注意力机制可以减少输入数据中的冗余信息，提高模型对关键特征的关注度。这有助于降低模型的计算复杂度，并提高模型的泛化能力。
3.  提升模型性能：加入CA注意力机制可以显著提高模型在多通道输入数据上的性能。它能够帮助模型更好地捕捉到通道之间的相关性和依赖关系，从而提高模型对输入数据的理解能力。

综上所述，加入CA注意力机制可以有效地增强模型对多通道输入数据的建模能力，提高模型性能和泛化能力。它在图像处理、视频分析等任务中具有重要的应用价值。✅

![](https://i-blog.csdnimg.cn/blog_migrate/fc17b013aaf3966056c9235e4dac7e5b.png)

> 论文题目：《Coordinate Attention for Efficient Mobile Network Design》
> 
> 论文地址： [https://arxiv.org/abs/2103.02907][https_arxiv.org_abs_2103.02907]
> 
> 代码实现： [houqb/CoordAttention首页 - GitCode][houqb_CoordAttention_ - GitCode]

#### 🚀2.网络结构 

本文的改进是基于YOLOv8，关于其网络结构具体如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/49a5ec7fe32dc41b7b72c5e8bceeaf04.png)

> YOLOv8官方仓库地址：
> 
> [GitHub - ultralytics/ultralytics: NEW - YOLOv8 🚀 in PyTorch > ONNX > OpenVINO > CoreML > TFLite][GitHub - ultralytics_ultralytics_ NEW - YOLOv8 _ in PyTorch _ ONNX _ OpenVINO _ CoreML _ TFLite]

#### 🚀3.添加步骤 

针对本文的改进，具体步骤如下所示：👇

步骤1：conv.py文件修改

步骤2：\_\_init\_\_.py文件修改

步骤3：tasks.py文件修改

步骤4：创建自定义yaml文件

步骤5：新建train.py文件

步骤6：模型训练测试

#### 🚀4.改进方法 

##### 🍀🍀步骤1：conv.py文件修改 

在源码中找到conv.py文件，具体位置是ultralytics/nn/modules/conv.py，然后将CA注意力机制模块代码添加到conv.py文件末尾位置。

CA注意力机制模块代码：

```java
# CA注意力机制代码
# By CSDN 小哥谈
import torch
import torch.nn as nn
import torch.nn.functional as F


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out
```

##### 🍀🍀步骤2：\_\_init\_\_.py文件修改 

在源码中找到\_\_init\_\_.py文件，具体位置是ultralytics/nn/modules/\_\_init\_\_.py。

修改1：加入CoordAtt，具体如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/577cdea6e958c7c615d40051295f96ca.png)

修改2：加入CoordAtt，具体如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/5216fc6f0a98421a1fb5f77388d84b9f.png)

##### 🍀🍀步骤3：tasks.py文件修改 

在源码中找到tasks.py文件，具体位置是ultralytics/nn/tasks.py。

修改1：在下图所示位置导入类名CoordAtt。

![](https://i-blog.csdnimg.cn/blog_migrate/116f57df376be6ab4e2b88b7a3791977.png)

修改2：找到parse\_model函数（736行左右），在下图中所示位置添加如下代码。

```java
# -----CA注意力机制--------
        elif m in {CoordAtt}:
            args=[ch[f],*args]
        # --------end------------
```

具体添加位置如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/9417be36e53c3bf92aed584d32b47f66.png)

##### 🍀🍀步骤4：创建自定义yaml文件 

在源码ultralytics/cfg/models/v8目录下创建yaml文件，并命名为：yolov8\_CA.yaml。具体如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/d7026b57c9d33e49d3ef5395910548ce.png)

yolov8\_CA.yaml文件完整代码如下所示：

```java
# Ultralytics YOLO 🚀, AGPL-3.0 license
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
  - [-1,1,CoordAtt,[]]
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
  - [-1, 6, C2f, [512, True]]
  - [-1,1,CoordAtt,[]]
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32
  - [-1, 3, C2f, [1024, True]]
  - [-1,1,CoordAtt,[]]
  - [-1, 1, SPPF, [1024, 5]]  # 9

# YOLOv8.0n head
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 8], 1, Concat, [1]]  # cat backbone P4
  - [-1, 3, C2f, [512]]  # 12

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 5], 1, Concat, [1]]  # cat backbone P3
  - [-1, 3, C2f, [256]]  # 15 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 15], 1, Concat, [1]]  # cat head P4
  - [-1, 3, C2f, [512]]  # 18 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]  # cat head P5
  - [-1, 3, C2f, [1024]]  # 21 (P5/32-large)

  - [[18, 21, 24], 1, Detect, [nc]]  # Detect(P3, P4, P5)
```

##### 🍀🍀步骤5：新建train.py文件 

在源码根目录下新建train.py文件，文件完整代码如下所示：

```java
from ultralytics import YOLO

# Load a model
model = YOLO(r'C:\Users\Lenovo\PycharmProjects\ultralytics-main\ultralytics\cfg\models\v8\yolov8_CA.yaml')  # build a new model from YAML
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
model = YOLO(r'C:\Users\Lenovo\PycharmProjects\ultralytics-main\ultralytics\cfg\models\v8\yolov8_CA.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
model.train(data=r'C:\Users\Lenovo\PycharmProjects\ultralytics-main\ultralytics\cfg\datasets\helmet.yaml', epochs=100, imgsz=640)
```

> 注意：一定要用绝对路径，以防发生报错。

##### 🍀🍀步骤6：模型训练测试 

在train.py文件，点击“运行”，在作者自制的安全帽佩戴检测数据集上，模型可以正常训练。

![](https://i-blog.csdnimg.cn/blog_migrate/98961de14bbbe99d57dc7731f513a37d.png)

模型训练过程： 

![](https://i-blog.csdnimg.cn/blog_migrate/c32c2d2f5de6aa1a1505e4a6869079a1.png)

模型训练结果：

![](https://i-blog.csdnimg.cn/blog_migrate/a255a588d980701a19a4a4fac396f1c7.png)

![](https://i-blog.csdnimg.cn/blog_migrate/a4634dd0fcdaf9907e748f112212cb80.png)

![](https://i-blog.csdnimg.cn/blog_migrate/df15c6d5127f81e046f41366ecab8007.gif)


[1.]: #%C2%A0%F0%9F%9A%801.%20%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5
[2.]: #%F0%9F%9A%802.%E6%B7%BB%E5%8A%A0%E4%BD%8D%E7%BD%AE
[3.]: #%F0%9F%9A%803.%E6%B7%BB%E5%8A%A0%E6%AD%A5%E9%AA%A4
[4.]: #%F0%9F%9A%804.%E6%94%B9%E8%BF%9B%E6%96%B9%E6%B3%95%C2%A0
[1_conv.py]: #%F0%9F%8D%80%F0%9F%8D%80%E6%8A%A5%E9%94%991
[2_init_.py]: #%F0%9F%8D%80%F0%9F%8D%80%E6%AD%A5%E9%AA%A42%EF%BC%9A__init__.py%E6%96%87%E4%BB%B6%E4%BF%AE%E6%94%B9
[3_tasks.py]: #%F0%9F%8D%80%F0%9F%8D%80%E6%AD%A5%E9%AA%A43%EF%BC%9Atasks.py%E6%96%87%E4%BB%B6%E4%BF%AE%E6%94%B9
[4_yaml]: #%F0%9F%8D%80%F0%9F%8D%80%E6%AD%A5%E9%AA%A44%EF%BC%9A%E5%88%9B%E5%BB%BA%E8%87%AA%E5%AE%9A%E4%B9%89yaml%E6%96%87%E4%BB%B6
[5_train.py]: #%F0%9F%8D%80%F0%9F%8D%80%E6%AD%A5%E9%AA%A45%EF%BC%9A%E6%96%B0%E5%BB%BAtrain.py%E6%96%87%E4%BB%B6
[6]: #%F0%9F%8D%80%F0%9F%8D%80%E6%AD%A5%E9%AA%A46%EF%BC%9A%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%E6%B5%8B%E8%AF%95
[https_arxiv.org_abs_2103.02907]: https://arxiv.org/abs/2103.02907
[houqb_CoordAttention_ - GitCode]: https://gitcode.com/houqb/CoordAttention/overview
[GitHub - ultralytics_ultralytics_ NEW - YOLOv8 _ in PyTorch _ ONNX _ OpenVINO _ CoreML _ TFLite]: https://github.com/ultralytics/ultralytics