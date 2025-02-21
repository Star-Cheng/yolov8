![](https://i-blog.csdnimg.cn/blog_migrate/98bc05efe779f503cc2eab7fa68fe1cc.jpeg)

> 前言：Hello大家好，我是小哥谈。DAT（Vision Transformer with Deformable Attention）是一种引入了可变形注意力机制的视觉Transformer。在训练算法模型的时候，通过引入可变形注意力机制，改进了视觉Transformer的效率和性能，使其在处理复杂的视觉任务时更加高效和准确。🌈 

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

![](https://i-blog.csdnimg.cn/blog_migrate/b85b93b3742db70368cb85a3b0a349d0.gif)

#### 🚀1.基础概念 

DAT（Vision Transformer with Deformable Attention）是一种引入了可变形注意力机制的视觉Transformer，DAT的核心思想主要包括以下几个方面：

1.  可变形注意力（Deformable Attention）：传统的Transformer使用标准的自注意力机制，这种机制会处理图像中的所有像素，导致计算量很大。而DAT引入了可变形注意力机制，它只关注图像中的一小部分关键区域。这种方法可以显著减少计算量，同时保持良好的性能。
2.  动态采样点：在可变形注意力机制中，DAT动态地选择采样点，而不是固定地处理整个图像。这种动态选择机制使得模型可以更加集中地关注于那些对当前任务最重要的区域。
3.  即插即用：DAT的设计允许它适应不同的图像大小和内容，使其在多种视觉任务中都能有效工作，如图像分类、对象检测等。

总结：DAT通过引入可变形注意力机制，改进了视觉Transformer的效率和性能，使其在处理复杂的视觉任务时更加高效和准确。

DAT的网络结构图：

![](https://i-blog.csdnimg.cn/blog_migrate/421a43353c9018096b1cfc6a0830818b.png)

![](https://i-blog.csdnimg.cn/blog_migrate/01372d1309223fe6e9470c8d46d7a65b.png)

> 论文题目：《Vision Transformer with Deformable Attention》
> 
> 论文地址： [https://arxiv.org/abs/2201.00520][https_arxiv.org_abs_2201.00520]
> 
> 代码实现：  [GitHub - LeapLabTHU/DAT: Repository of Vision Transformer with Deformable Attention (CVPR2022) and DAT++: Spatially Dynamic Vision Transformerwith Deformable Attention][GitHub - LeapLabTHU_DAT_ Repository of Vision Transformer with Deformable Attention _CVPR2022_ and DAT_ Spatially Dynamic Vision Transformerwith Deformable Attention]

#### 🚀2.网络结构 

本文的改进是基于YOLOv8，关于其网络结构具体如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/29a1a714cfc2deaed3ac3f5be484337e.png)

> YOLOv8官方仓库地址：
> 
> [GitHub - ultralytics/ultralytics: NEW - YOLOv8 🚀 in PyTorch > ONNX > OpenVINO > CoreML > TFLite][GitHub - ultralytics_ultralytics_ NEW - YOLOv8 _ in PyTorch _ ONNX _ OpenVINO _ CoreML _ TFLite]

本文所作的改进是在YOLOv8中添加DAT注意力机制。改进后的完整网络结构图如下所示：

![](https://i-blog.csdnimg.cn/blog_migrate/836e858fa777137ba3bad3c2af9a4b25.png)

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

在源码中找到conv.py文件，具体位置是ultralytics/nn/modules/conv.py，然后将DAT注意力机制模块代码添加到conv.py文件末尾位置。

DAT注意力机制模块代码：

```java
# DAT注意力模块
# By CSDN 小哥谈
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from timm.models.layers import to_2tuple, trunc_normal_

class LayerNormProxy(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')


class DAT(nn.Module):

    def __init__(
            self, q_size=(224,224), kv_size=(224,224), n_heads=8, n_head_channels=32, n_groups=1,
            attn_drop=0.0, proj_drop=0.0, stride=1,
            offset_range_factor=-1, use_pe=True, dwc_pe=True,
            no_off=False, fixed_pe=False, ksize=9, log_cpb=False
    ):

        super().__init__()
        n_head_channels = int(q_size / 8)
        q_size = (q_size, q_size)

        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        # self.kv_h, self.kv_w = kv_size
        self.kv_h, self.kv_w = self.q_h // stride, self.q_w // stride
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        self.ksize = ksize
        self.log_cpb = log_cpb
        self.stride = stride
        kk = self.ksize
        pad_size = kk // 2 if kk != stride else 0


        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )

        if self.no_off:
            for m in self.conv_offset.parameters():
                m.requires_grad_(False)

        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0)

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )
        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        if self.use_pe and not self.no_off:
            if self.dwc_pe:
                self.rpe_table = nn.Conv2d(
                    self.nc, self.nc, kernel_size=3, stride=1, padding=1, groups=self.nc)
            elif self.fixed_pe:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * self.q_w, self.kv_h * self.kv_w)
                )
                trunc_normal_(self.rpe_table, std=0.01)
            elif self.log_cpb:
                # Borrowed from Swin-V2
                self.rpe_table = nn.Sequential(
                    nn.Linear(2, 32, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, self.n_group_heads, bias=False)
                )
            else:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * 2 - 1, self.q_w * 2 - 1)
                )
                trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2

        return ref

    @torch.no_grad()
    def _get_q_grid(self, H, W, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.arange(0, H, dtype=dtype, device=device),
            torch.arange(0, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2

        return ref

    def forward(self, x):
        x = x
        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        q = self.proj_q(x)
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off).contiguous()  # B * g 2 Hg Wg
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk

        if self.offset_range_factor >= 0 and not self.no_off:
            offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        if self.no_off:
            offset = offset.fill_(0.0)

        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).clamp(-1., +1.)

        if self.no_off:
            x_sampled = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
            assert x_sampled.size(2) == Hk and x_sampled.size(3) == Wk, f"Size is {x_sampled.size()}"
        else:
            x_sampled = F.grid_sample(
                input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
                grid=pos[..., (1, 0)],  # y, x -> x, y
                mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg

        x_sampled = x_sampled.reshape(B, C, 1, n_sample)
        # self.proj_k.weight = torch.nn.Parameter(self.proj_k.weight.float())
        # self.proj_k.bias = torch.nn.Parameter(self.proj_k.bias.float())
        # self.proj_v.weight = torch.nn.Parameter(self.proj_v.weight.float())
        # self.proj_v.bias = torch.nn.Parameter(self.proj_v.bias.float())
        # 检查权重的数据类型
        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)

        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        attn = attn.mul(self.scale)

        if self.use_pe and (not self.no_off):

            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * self.n_heads, self.n_head_channels,
                                                                              H * W)
            elif self.fixed_pe:
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                attn = attn + attn_bias.reshape(B * self.n_heads, H * W, n_sample)
            elif self.log_cpb:
                q_grid = self._get_q_grid(H, W, B, dtype, device)
                displacement = (
                            q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups,
                                                                                                   n_sample,
                                                                                                   2).unsqueeze(1)).mul(
                    4.0)  # d_y, d_x [-8, +8]
                displacement = torch.sign(displacement) * torch.log2(torch.abs(displacement) + 1.0) / np.log2(8.0)
                attn_bias = self.rpe_table(displacement)  # B * g, H * W, n_sample, h_g
                attn = attn + einops.rearrange(attn_bias, 'b m n h -> (b h) m n', h=self.n_group_heads)
            else:
                rpe_table = self.rpe_table
                rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                q_grid = self._get_q_grid(H, W, B, dtype, device)
                displacement = (
                            q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups,
                                                                                                   n_sample,
                                                                                                   2).unsqueeze(1)).mul(
                    0.5)
                attn_bias = F.grid_sample(
                    input=einops.rearrange(rpe_bias, 'b (g c) h w -> (b g) c h w', c=self.n_group_heads,
                                           g=self.n_groups),
                    grid=displacement[..., (1, 0)],
                    mode='bilinear', align_corners=True)  # B * g, h_g, HW, Ns

                attn_bias = attn_bias.reshape(B * self.n_heads, H * W, n_sample)
                attn = attn + attn_bias

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b c n -> b c m', attn, v)

        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe
        out = out.reshape(B, C, H, W)

        y = self.proj_drop(self.proj_out(out))
        h, w = pos.reshape(B, self.n_groups, Hk, Wk, 2), reference.reshape(B, self.n_groups, Hk, Wk, 2)

        return y
```

再然后，在block.py文件最上方下图所示位置加入DAT。

![](https://i-blog.csdnimg.cn/blog_migrate/cf43deeb18b263a2e59a498f064b4a2a.png)

##### 🍀🍀步骤2：\_\_init\_\_.py文件修改 

在源码中找到\_\_init\_\_.py文件，具体位置是ultralytics/nn/modules/\_\_init\_\_.py。

修改1：加入DAT，具体如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/f6c00e6d215c0e9121aedd51914a6a2d.png)

修改2：加入DAT，具体如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/0c24be70c5b17f1a42e98578371b2c43.png)

##### 🍀🍀步骤3：tasks.py文件修改 

在源码中找到tasks.py文件，具体位置是ultralytics/nn/tasks.py。

修改1：在下图所示位置加入DAT。

![](https://i-blog.csdnimg.cn/blog_migrate/9ffe935eae650bebb6d85229eeffe1c0.png)

修改2：找到parse\_model函数（736行左右），在下图中所示位置加入代码。

```java
# -----DAT注意力机制--------
        elif m in {DAT}:
            args = [ch[f], *args]
        # --------end------------
```

具体添加位置如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/d2c63ca6403610fd729eefdbecc0f2ef.png)

##### 🍀🍀步骤4：创建自定义yaml文件 

在源码ultralytics/cfg/models/v8目录下创建yaml文件，并命名为：yolov8\_DAT.yaml。具体如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/04f0e551699eaafaa02675fad8803e60.png)

yolov8\_DAT.yaml文件完整代码如下所示：👇

```java
# Ultralytics YOLO 🚀, AGPL-3.0 license
# YOLOv8 object detection model with P3-P5 outputs. For Usage examples see https://docs.ultralytics.com/tasks/detect

# Parameters
nc: 80 # number of classes
scales: # model compound scaling constants, i.e. 'model=yolov8n.yaml' will call yolov8.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.33, 0.25, 1024] # YOLOv8n summary: 225 layers,  3157200 parameters,  3157184 gradients,   8.9 GFLOPs
  s: [0.33, 0.50, 1024] # YOLOv8s summary: 225 layers, 11166560 parameters, 11166544 gradients,  28.8 GFLOPs
  m: [0.67, 0.75, 768] # YOLOv8m summary: 295 layers, 25902640 parameters, 25902624 gradients,  79.3 GFLOPs
  l: [1.00, 1.00, 512] # YOLOv8l summary: 365 layers, 43691520 parameters, 43691504 gradients, 165.7 GFLOPs
  x: [1.00, 1.25, 512] # YOLOv8x summary: 365 layers, 68229648 parameters, 68229632 gradients, 258.5 GFLOPs

# YOLOv8.0n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]] # 9

# YOLOv8.0n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 3, C2f, [512]] # 12

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 3, C2f, [256]] # 15 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]] # cat head P4
  - [-1, 3, C2f, [512]] # 18 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]] # cat head P5
  - [-1, 3, C2f, [1024]] # 21 (P5/32-large)
  - [-1, 1, DAT, []]

  - [[15, 18, 22], 1, Detect, [nc]] # Detect(P3, P4, P5)
```

##### 🍀🍀步骤5：新建train.py文件 

在源码根目录下新建train.py文件，文件完整代码如下所示：

```java
from ultralytics import YOLO

# Load a model
model = YOLO(r'C:\Users\Lenovo\PycharmProjects\ultralytics-main\ultralytics\cfg\models\v8\yolov8_DAT.yaml')  # build a new model from YAML
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
model = YOLO(r'C:\Users\Lenovo\PycharmProjects\ultralytics-main\ultralytics\cfg\models\v8\yolov8_DAT.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
model.train(data=r'C:\Users\Lenovo\PycharmProjects\ultralytics-main\ultralytics\cfg\datasets\helmet.yaml', epochs=100,imgsz=640)
```

> 注意：一定要用绝对路径，以防发生报错。

##### 🍀🍀步骤6：模型训练测试 

在train.py文件，点击“运行”，在作者自制的安全帽佩戴检测数据集上，模型可以正常训练。

![](https://i-blog.csdnimg.cn/blog_migrate/607b036b6e66713764031885c738c1e9.png)

> 参数量对比：🌈
> 
> yolov8.yaml：225 layers, 3011238 parameters, 3011222 gradients, 8.2 GFLOPs
> 
> yolov8\_DAT.yaml：239 layers, 3298982 parameters, 3298966 gradients, 8.4 GFLOPs

模型训练结果：

![](https://i-blog.csdnimg.cn/blog_migrate/03e70b1a5e0de4052be9e19689a253ec.png)

![](https://i-blog.csdnimg.cn/blog_migrate/9f11403bacc174a04cd8674757323458.gif)


[1.]: #%F0%9F%9A%801.%20%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5
[2.]: #%F0%9F%9A%802.%E6%B7%BB%E5%8A%A0%E4%BD%8D%E7%BD%AE
[3.]: #%F0%9F%9A%803.%E6%B7%BB%E5%8A%A0%E6%AD%A5%E9%AA%A4
[4.]: #%F0%9F%9A%804.%E6%94%B9%E8%BF%9B%E6%96%B9%E6%B3%95%C2%A0
[1_conv.py]: #%F0%9F%8D%80%F0%9F%8D%80%E6%8A%A5%E9%94%991
[2_init_.py]: #%F0%9F%8D%80%F0%9F%8D%80%E6%AD%A5%E9%AA%A42%EF%BC%9A__init__.py%E6%96%87%E4%BB%B6%E4%BF%AE%E6%94%B9
[3_tasks.py]: #%F0%9F%8D%80%F0%9F%8D%80%E6%AD%A5%E9%AA%A43%EF%BC%9Atasks.py%E6%96%87%E4%BB%B6%E4%BF%AE%E6%94%B9
[4_yaml]: #%F0%9F%8D%80%F0%9F%8D%80%E6%AD%A5%E9%AA%A44%EF%BC%9A%E5%88%9B%E5%BB%BA%E8%87%AA%E5%AE%9A%E4%B9%89yaml%E6%96%87%E4%BB%B6
[5_train.py]: #%F0%9F%8D%80%F0%9F%8D%80%E6%AD%A5%E9%AA%A45%EF%BC%9A%E6%96%B0%E5%BB%BAtrain.py%E6%96%87%E4%BB%B6
[6]: #%F0%9F%8D%80%F0%9F%8D%80%E6%AD%A5%E9%AA%A46%EF%BC%9A%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%E6%B5%8B%E8%AF%95
[https_arxiv.org_abs_2201.00520]: https://arxiv.org/abs/2201.00520
[GitHub - LeapLabTHU_DAT_ Repository of Vision Transformer with Deformable Attention _CVPR2022_ and DAT_ Spatially Dynamic Vision Transformerwith Deformable Attention]: https://github.com/LeapLabTHU/DAT
[GitHub - ultralytics_ultralytics_ NEW - YOLOv8 _ in PyTorch _ ONNX _ OpenVINO _ CoreML _ TFLite]: https://github.com/ultralytics/ultralytics