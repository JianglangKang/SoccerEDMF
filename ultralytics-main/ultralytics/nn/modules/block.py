# # # Ultralytics YOLO 🚀, AGPL-3.0 license
# # """Block modules."""
# # from typing import Tuple
# #
# # from einops import rearrange
# # from torch import Tensor
#
# # Ultralytics YOLO 🚀, AGPL-3.0 license
# """Block modules."""
# from typing import Tuple
#
# from einops import rearrange
# from torch import Tensor
#
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# from .rep_block import *
# from .attention import *
# from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
# from .transformer import TransformerBlock
#
# __all__ = (
#     "DFL",
#     "HGBlock",
#     "HGStem",
#     "SPP",
#     "SPPF",
#     "C1",
#     "C2",
#     "C3",
#     "C2f",
#     "C2fAttn",
#     "ImagePoolingAttn",
#     "ContrastiveHead",
#     "BNContrastiveHead",
#     "C3x",
#     "C3TR",
#     "C3Ghost",
#     "GhostBottleneck",
#     "Bottleneck",
#     "BottleneckCSP",
#     "Proto",
#     "RepC3",
#     "ResNetLayer",
#     "RepNCSPELAN4",
#     "ADown",
#     "SPPELAN",
#     "CBFuse",
#     "CBLinear",
#     "Silence",
#     "BiLevelRoutingAttention",
#     'RepBlock',
#     'ScalSeq',
#     'Add',
#     'Zoom_cat',
#     'C2f_Faster',
#     'C2f_Faster_EMA',
#     'C2f_DBB',
#     'CSPStage',
#     'Fusion',
#     'FocusFeature',
#
# )
#
#
# class DFL(nn.Module):
#     """
#     Integral module of Distribution Focal Loss (DFL).
#
#     Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
#     """
#
#     def __init__(self, c1=16):
#         """Initialize a convolutional layer with a given number of input channels."""
#         super().__init__()
#         self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
#         x = torch.arange(c1, dtype=torch.float)
#         self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
#         self.c1 = c1
#
#     def forward(self, x):
#         """Applies a transformer layer on input tensor 'x' and returns a tensor."""
#         b, _, a = x.shape  # batch, channels, anchors
#         return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
#         # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)
#
#
# class Proto(nn.Module):
#     """YOLOv8 mask Proto module for segmentation models."""
#
#     def __init__(self, c1, c_=256, c2=32):
#         """
#         Initializes the YOLOv8 mask Proto module with specified number of protos and masks.
#
#         Input arguments are ch_in, number of protos, number of masks.
#         """
#         super().__init__()
#         self.cv1 = Conv(c1, c_, k=3)
#         self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
#         self.cv2 = Conv(c_, c_, k=3)
#         self.cv3 = Conv(c_, c2)
#
#     def forward(self, x):
#         """Performs a forward pass through layers using an upsampled input image."""
#         return self.cv3(self.cv2(self.upsample(self.cv1(x))))
#
#
# class HGStem(nn.Module):
#     """
#     StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.
#
#     https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
#     """
#
#     def __init__(self, c1, cm, c2):
#         """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
#         super().__init__()
#         self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
#         self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
#         self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
#         self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
#         self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)
#
#     def forward(self, x):
#         """Forward pass of a PPHGNetV2 backbone layer."""
#         x = self.stem1(x)
#         x = F.pad(x, [0, 1, 0, 1])
#         x2 = self.stem2a(x)
#         x2 = F.pad(x2, [0, 1, 0, 1])
#         x2 = self.stem2b(x2)
#         x1 = self.pool(x)
#         x = torch.cat([x1, x2], dim=1)
#         x = self.stem3(x)
#         x = self.stem4(x)
#         return x
#
#
# class HGBlock(nn.Module):
#     """
#     HG_Block of PPHGNetV2 with 2 convolutions and LightConv.
#
#     https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
#     """
#
#     def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
#         """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
#         super().__init__()
#         block = LightConv if lightconv else Conv
#         self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
#         self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
#         self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
#         self.add = shortcut and c1 == c2
#
#     def forward(self, x):
#         """Forward pass of a PPHGNetV2 backbone layer."""
#         y = [x]
#         y.extend(m(y[-1]) for m in self.m)
#         y = self.ec(self.sc(torch.cat(y, 1)))
#         return y + x if self.add else y
#
#
# class SPP(nn.Module):
#     """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""
#
#     def __init__(self, c1, c2, k=(5, 9, 13)):
#         """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
#         super().__init__()
#         c_ = c1 // 2  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
#         self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
#
#     def forward(self, x):
#         """Forward pass of the SPP layer, performing spatial pyramid pooling."""
#         x = self.cv1(x)
#         return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))
#
#
# class SPPF(nn.Module):
#     """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""
#
#     def __init__(self, c1, c2, k=5):
#         """
#         Initializes the SPPF layer with given input/output channels and kernel size.
#
#         This module is equivalent to SPP(k=(5, 9, 13)).
#         """
#         super().__init__()
#         c_ = c1 // 2  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c_ * 4, c2, 1, 1)
#         self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#
#     def forward(self, x):
#         """Forward pass through Ghost Convolution block."""
#         y = [self.cv1(x)]
#         y.extend(self.m(y[-1]) for _ in range(3))
#         return self.cv2(torch.cat(y, 1))
#
#
# class C1(nn.Module):
#     """CSP Bottleneck with 1 convolution."""
#
#     def __init__(self, c1, c2, n=1):
#         """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
#         super().__init__()
#         self.cv1 = Conv(c1, c2, 1, 1)
#         self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))
#
#     def forward(self, x):
#         """Applies cross-convolutions to input in the C3 module."""
#         y = self.cv1(x)
#         return self.m(y) + y
#
#
# class C2(nn.Module):
#     """CSP Bottleneck with 2 convolutions."""
#
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         """Initializes the CSP Bottleneck with 2 convolutions module with arguments ch_in, ch_out, number, shortcut,
#         groups, expansion.
#         """
#         super().__init__()
#         self.c = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
#         self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
#         # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
#         self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))
#
#     def forward(self, x):
#         """Forward pass through the CSP bottleneck with 2 convolutions."""
#         a, b = self.cv1(x).chunk(2, 1)
#         return self.cv2(torch.cat((self.m(a), b), 1))
#
#
# class C2f(nn.Module):
#     """Faster Implementation of CSP Bottleneck with 2 convolutions."""
#
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
#         """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
#         expansion.
#         """
#         super().__init__()
#         self.c = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
#         self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
#         self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
#
#     def forward(self, x):
#         """Forward pass through C2f layer."""
#         y = list(self.cv1(x).chunk(2, 1))
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(torch.cat(y, 1))
#
#     def forward_split(self, x):
#         """Forward pass using split() instead of chunk()."""
#         y = list(self.cv1(x).split((self.c, self.c), 1))
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(torch.cat(y, 1))
#
#
# class C3(nn.Module):
#     """CSP Bottleneck with 3 convolutions."""
#
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c1, c_, 1, 1)
#         self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
#         self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))
#
#     def forward(self, x):
#         """Forward pass through the CSP bottleneck with 2 convolutions."""
#         return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
#
#
# class C3x(C3):
#     """C3 module with cross-convolutions."""
#
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         """Initialize C3TR instance and set default parameters."""
#         super().__init__(c1, c2, n, shortcut, g, e)
#         self.c_ = int(c2 * e)
#         self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))
#
#
# class RepC3(nn.Module):
#     """Rep C3."""
#
#     def __init__(self, c1, c2, n=3, e=1.0):
#         """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c2, 1, 1)
#         self.cv2 = Conv(c1, c2, 1, 1)
#         self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
#         self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()
#
#     def forward(self, x):
#         """Forward pass of RT-DETR neck layer."""
#         return self.cv3(self.m(self.cv1(x)) + self.cv2(x))
#
#
# class C3TR(C3):
#     """C3 module with TransformerBlock()."""
#
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         """Initialize C3Ghost module with GhostBottleneck()."""
#         super().__init__(c1, c2, n, shortcut, g, e)
#         c_ = int(c2 * e)
#         self.m = TransformerBlock(c_, c_, 4, n)
#
#
# class C3Ghost(C3):
#     """C3 module with GhostBottleneck()."""
#
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
#         super().__init__(c1, c2, n, shortcut, g, e)
#         c_ = int(c2 * e)  # hidden channels
#         self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))
#
#
# class GhostBottleneck(nn.Module):
#     """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""
#
#     def __init__(self, c1, c2, k=3, s=1):
#         """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
#         super().__init__()
#         c_ = c2 // 2
#         self.conv = nn.Sequential(
#             GhostConv(c1, c_, 1, 1),  # pw
#             DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
#             GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
#         )
#         self.shortcut = (
#             nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
#         )
#
#     def forward(self, x):
#         """Applies skip connection and concatenation to input tensor."""
#         return self.conv(x) + self.shortcut(x)
#
#
# class Bottleneck(nn.Module):
#     """Standard bottleneck."""
#
#     def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
#         """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
#         expansion.
#         """
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, k[0], 1)
#         self.cv2 = Conv(c_, c2, k[1], 1, g=g)
#         self.add = shortcut and c1 == c2
#
#     def forward(self, x):
#         """'forward()' applies the YOLO FPN to input data."""
#         return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
#
#
# class BottleneckCSP(nn.Module):
#     """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""
#
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
#         self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
#         self.cv4 = Conv(2 * c_, c2, 1, 1)
#         self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
#         self.act = nn.SiLU()
#         self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
#
#     def forward(self, x):
#         """Applies a CSP bottleneck with 3 convolutions."""
#         y1 = self.cv3(self.m(self.cv1(x)))
#         y2 = self.cv2(x)
#         return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))
#
#
# class ResNetBlock(nn.Module):
#     """ResNet block with standard convolution layers."""
#
#     def __init__(self, c1, c2, s=1, e=4):
#         """Initialize convolution with given parameters."""
#         super().__init__()
#         c3 = e * c2
#         self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
#         self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
#         self.cv3 = Conv(c2, c3, k=1, act=False)
#         self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()
#
#     def forward(self, x):
#         """Forward pass through the ResNet block."""
#         return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))
#
#
# class ResNetLayer(nn.Module):
#     """ResNet layer with multiple ResNet blocks."""
#
#     def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
#         """Initializes the ResNetLayer given arguments."""
#         super().__init__()
#         self.is_first = is_first
#
#         if self.is_first:
#             self.layer = nn.Sequential(
#                 Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#             )
#         else:
#             blocks = [ResNetBlock(c1, c2, s, e=e)]
#             blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
#             self.layer = nn.Sequential(*blocks)
#
#     def forward(self, x):
#         """Forward pass through the ResNet layer."""
#         return self.layer(x)
#
#
# class MaxSigmoidAttnBlock(nn.Module):
#     """Max Sigmoid attention block."""
#
#     def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
#         """Initializes MaxSigmoidAttnBlock with specified arguments."""
#         super().__init__()
#         self.nh = nh
#         self.hc = c2 // nh
#         self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
#         self.gl = nn.Linear(gc, ec)
#         self.bias = nn.Parameter(torch.zeros(nh))
#         self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
#         self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0
#
#     def forward(self, x, guide):
#         """Forward process."""
#         bs, _, h, w = x.shape
#
#         guide = self.gl(guide)
#         guide = guide.view(bs, -1, self.nh, self.hc)
#         embed = self.ec(x) if self.ec is not None else x
#         embed = embed.view(bs, self.nh, self.hc, h, w)
#
#         aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
#         aw = aw.max(dim=-1)[0]
#         aw = aw / (self.hc**0.5)
#         aw = aw + self.bias[None, :, None, None]
#         aw = aw.sigmoid() * self.scale
#
#         x = self.proj_conv(x)
#         x = x.view(bs, self.nh, -1, h, w)
#         x = x * aw.unsqueeze(2)
#         return x.view(bs, -1, h, w)
#
#
# class C2fAttn(nn.Module):
#     """C2f module with an additional attn module."""
#
#     def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
#         """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
#         expansion.
#         """
#         super().__init__()
#         self.c = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
#         self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
#         self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
#         self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)
#
#     def forward(self, x, guide):
#         """Forward pass through C2f layer."""
#         y = list(self.cv1(x).chunk(2, 1))
#         y.extend(m(y[-1]) for m in self.m)
#         y.append(self.attn(y[-1], guide))
#         return self.cv2(torch.cat(y, 1))
#
#     def forward_split(self, x, guide):
#         """Forward pass using split() instead of chunk()."""
#         y = list(self.cv1(x).split((self.c, self.c), 1))
#         y.extend(m(y[-1]) for m in self.m)
#         y.append(self.attn(y[-1], guide))
#         return self.cv2(torch.cat(y, 1))
#
#
# class ImagePoolingAttn(nn.Module):
#     """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""
#
#     def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
#         """Initializes ImagePoolingAttn with specified arguments."""
#         super().__init__()
#
#         nf = len(ch)
#         self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
#         self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
#         self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
#         self.proj = nn.Linear(ec, ct)
#         self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
#         self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
#         self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
#         self.ec = ec
#         self.nh = nh
#         self.nf = nf
#         self.hc = ec // nh
#         self.k = k
#
#     def forward(self, x, text):
#         """Executes attention mechanism on input tensor x and guide tensor."""
#         bs = x[0].shape[0]
#         assert len(x) == self.nf
#         num_patches = self.k**2
#         x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
#         x = torch.cat(x, dim=-1).transpose(1, 2)
#         q = self.query(text)
#         k = self.key(x)
#         v = self.value(x)
#
#         # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
#         q = q.reshape(bs, -1, self.nh, self.hc)
#         k = k.reshape(bs, -1, self.nh, self.hc)
#         v = v.reshape(bs, -1, self.nh, self.hc)
#
#         aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
#         aw = aw / (self.hc**0.5)
#         aw = F.softmax(aw, dim=-1)
#
#         x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
#         x = self.proj(x.reshape(bs, -1, self.ec))
#         return x * self.scale + text
#
#
# class ContrastiveHead(nn.Module):
#     """Contrastive Head for YOLO-World compute the region-text scores according to the similarity between image and text
#     features.
#     """
#
#     def __init__(self):
#         """Initializes ContrastiveHead with specified region-text similarity parameters."""
#         super().__init__()
#         # NOTE: use -10.0 to keep the init cls loss consistency with other losses
#         self.bias = nn.Parameter(torch.tensor([-10.0]))
#         self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())
#
#     def forward(self, x, w):
#         """Forward function of contrastive learning."""
#         x = F.normalize(x, dim=1, p=2)
#         w = F.normalize(w, dim=-1, p=2)
#         x = torch.einsum("bchw,bkc->bkhw", x, w)
#         return x * self.logit_scale.exp() + self.bias
#
#
# class BNContrastiveHead(nn.Module):
#     """
#     Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.
#
#     Args:
#         embed_dims (int): Embed dimensions of text and image features.
#     """
#
#     def __init__(self, embed_dims: int):
#         """Initialize ContrastiveHead with region-text similarity parameters."""
#         super().__init__()
#         self.norm = nn.BatchNorm2d(embed_dims)
#         # NOTE: use -10.0 to keep the init cls loss consistency with other losses
#         self.bias = nn.Parameter(torch.tensor([-10.0]))
#         # use -1.0 is more stable
#         self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))
#
#     def forward(self, x, w):
#         """Forward function of contrastive learning."""
#         x = self.norm(x)
#         w = F.normalize(w, dim=-1, p=2)
#         x = torch.einsum("bchw,bkc->bkhw", x, w)
#         return x * self.logit_scale.exp() + self.bias
#
#
# class RepBottleneck(Bottleneck):
#     """Rep bottleneck."""
#
#     def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
#         """Initializes a RepBottleneck module with customizable in/out channels, shortcut option, groups and expansion
#         ratio.
#         """
#         super().__init__(c1, c2, shortcut, g, k, e)
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = RepConv(c1, c_, k[0], 1)
#
#
# class RepCSP(C3):
#     """Rep CSP Bottleneck with 3 convolutions."""
#
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
#         super().__init__(c1, c2, n, shortcut, g, e)
#         c_ = int(c2 * e)  # hidden channels
#         self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
#
#
# class RepNCSPELAN4(nn.Module):
#     """CSP-ELAN."""
#
#     def __init__(self, c1, c2, c3, c4, n=1):
#         """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
#         super().__init__()
#         self.c = c3 // 2
#         self.cv1 = Conv(c1, c3, 1, 1)
#         self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
#         self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
#         self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)
#
#     def forward(self, x):
#         """Forward pass through RepNCSPELAN4 layer."""
#         y = list(self.cv1(x).chunk(2, 1))
#         y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
#         return self.cv4(torch.cat(y, 1))
#
#     def forward_split(self, x):
#         """Forward pass using split() instead of chunk()."""
#         y = list(self.cv1(x).split((self.c, self.c), 1))
#         y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
#         return self.cv4(torch.cat(y, 1))
#
# '''添加代码'''
# # c2f
# from timm.models.layers import DropPath
#
#
# class Partial_conv3(nn.Module):
#     def __init__(self, dim, n_div=4, forward='split_cat'):
#         super().__init__()
#         self.dim_conv3 = dim // n_div
#         self.dim_untouched = dim - self.dim_conv3
#         self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
#
#         if forward == 'slicing':
#             self.forward = self.forward_slicing
#         elif forward == 'split_cat':
#             self.forward = self.forward_split_cat
#         else:
#             raise NotImplementedError
#
#     def forward_slicing(self, x):
#         # only for inference
#         x = x.clone()  # !!! Keep the original input intact for the residual connection later
#         x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
#         return x
#
#     def forward_split_cat(self, x):
#         # for training/inference
#         x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
#         x1 = self.partial_conv3(x1)
#         x = torch.cat((x1, x2), 1)
#         return x
#
#
# class Faster_Block(nn.Module):
#     def __init__(self,
#                  inc,
#                  dim,
#                  n_div=4,
#                  mlp_ratio=2,
#                  drop_path=0.1,
#                  layer_scale_init_value=0.0,
#                  pconv_fw_type='split_cat'
#                  ):
#         super().__init__()
#         self.dim = dim
#         self.mlp_ratio = mlp_ratio
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.n_div = n_div
#
#         mlp_hidden_dim = int(dim * mlp_ratio)
#
#         mlp_layer = [
#             Conv(dim, mlp_hidden_dim, 1),
#             nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
#         ]
#
#         self.mlp = nn.Sequential(*mlp_layer)
#
#         self.spatial_mixing = Partial_conv3(
#             dim,
#             n_div,
#             pconv_fw_type
#         )
#
#         self.adjust_channel = None
#         if inc != dim:
#             self.adjust_channel = Conv(inc, dim, 1)
#
#         if layer_scale_init_value > 0:
#             self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
#             self.forward = self.forward_layer_scale
#         else:
#             self.forward = self.forward
#
#     def forward(self, x):
#         if self.adjust_channel is not None:
#             x = self.adjust_channel(x)
#         shortcut = x
#         x = self.spatial_mixing(x)
#         x = shortcut + self.drop_path(self.mlp(x))
#         return x
#
#     def forward_layer_scale(self, x):
#         shortcut = x
#         x = self.spatial_mixing(x)
#         x = shortcut + self.drop_path(
#             self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
#         return x
#
#
# class C3_Faster(C3):
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
#         super().__init__(c1, c2, n, shortcut, g, e)
#         c_ = int(c2 * e)  # hidden channels
#         self.m = nn.Sequential(*(Faster_Block(c_, c_) for _ in range(n)))
#
#
# class C2f_Faster(C2f):
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
#         super().__init__(c1, c2, n, shortcut, g, e)
#         self.m = nn.ModuleList(Faster_Block(self.c, self.c) for _ in range(n))
#
#
# # repblock
# def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
#     result = nn.Sequential()
#     result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
#                                                   kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
#     result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
#     return result
# class RepVGGBlock(nn.Module):
#
#     def __init__(self, in_channels, out_channels, kernel_size=3,
#                  stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
#         super(RepVGGBlock, self).__init__()
#         self.deploy = deploy
#         self.groups = groups
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         assert kernel_size == 3
#         assert padding == 1
#
#         padding_11 = padding - kernel_size // 2
#
#         self.nonlinearity = nn.ReLU()
#
#         if use_se:
#             #   Note that RepVGG-D2se uses SE before nonlinearity. But RepVGGplus models uses SE after nonlinearity.
#             # self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
#             raise  NotImplementedError('se block not supported')
#         else:
#             self.se = nn.Identity()
#
#         if deploy:
#             self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
#                                       padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
#
#         else:
#             self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
#             self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
#             self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)
#             print('RepVGG Block, identity = ', self.rbr_identity)
#
#
#     def forward(self, inputs):
#         if hasattr(self, 'rbr_reparam'):
#             return self.nonlinearity(self.se(self.rbr_reparam(inputs)))
#
#         if self.rbr_identity is None:
#             id_out = 0
#         else:
#             id_out = self.rbr_identity(inputs)
#
#         return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))
#
#
#     #   Optional. This may improve the accuracy and facilitates quantization in some cases.
#     #   1.  Cancel the original weight decay on rbr_dense.conv.weight and rbr_1x1.conv.weight.
#     #   2.  Use like this.
#     #       loss = criterion(....)
#     #       for every RepVGGBlock blk:
#     #           loss += weight_decay_coefficient * 0.5 * blk.get_cust_L2()
#     #       optimizer.zero_grad()
#     #       loss.backward()
#     def get_custom_L2(self):
#         K3 = self.rbr_dense.conv.weight
#         K1 = self.rbr_1x1.conv.weight
#         t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
#         t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
#
#         l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()      # The L2 loss of the "circle" of weights in 3x3 kernel. Use regular L2 on them.
#         eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1                           # The equivalent resultant central point of 3x3 kernel.
#         l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()        # Normalize for an L2 coefficient comparable to regular L2.
#         return l2_loss_eq_kernel + l2_loss_circle
#
#
#
# #   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
# #   You can get the equivalent kernel and bias at any time and do whatever you want,
#     #   for example, apply some penalties or constraints during training, just like you do to the other models.
# #   May be useful for quantization or pruning.
#     def get_equivalent_kernel_bias(self):
#         kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
#         kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
#         kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
#         return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid
#
#     def _pad_1x1_to_3x3_tensor(self, kernel1x1):
#         if kernel1x1 is None:
#             return 0
#         else:
#             return torch.nn.functional.pad(kernel1x1, [1,1,1,1])
#
#     def _fuse_bn_tensor(self, branch):
#         if branch is None:
#             return 0, 0
#         if isinstance(branch, nn.Sequential):
#             kernel = branch.conv.weight
#             running_mean = branch.bn.running_mean
#             running_var = branch.bn.running_var
#             gamma = branch.bn.weight
#             beta = branch.bn.bias
#             eps = branch.bn.eps
#         else:
#             assert isinstance(branch, nn.BatchNorm2d)
#             if not hasattr(self, 'id_tensor'):
#                 input_dim = self.in_channels // self.groups
#                 kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
#                 for i in range(self.in_channels):
#                     kernel_value[i, i % input_dim, 1, 1] = 1
#                 self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
#             kernel = self.id_tensor
#             running_mean = branch.running_mean
#             running_var = branch.running_var
#             gamma = branch.weight
#             beta = branch.bias
#             eps = branch.eps
#         std = (running_var + eps).sqrt()
#         t = (gamma / std).reshape(-1, 1, 1, 1)
#         return kernel * t, beta - running_mean * gamma / std
#
#     def switch_to_deploy(self):
#         if hasattr(self, 'rbr_reparam'):
#             return
#         kernel, bias = self.get_equivalent_kernel_bias()
#         self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
#                                      kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
#                                      padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
#         self.rbr_reparam.weight.data = kernel
#         self.rbr_reparam.bias.data = bias
#         self.__delattr__('rbr_dense')
#         self.__delattr__('rbr_1x1')
#         if hasattr(self, 'rbr_identity'):
#             self.__delattr__('rbr_identity')
#         if hasattr(self, 'id_tensor'):
#             self.__delattr__('id_tensor')
#         self.deploy = True
#
# class BottleRep(nn.Module):
#     def __init__(self, in_channels, out_channels, basic_block=RepVGGBlock, weight=False):
#         super().__init__()
#         self.conv1 = basic_block(in_channels,out_channels)
#         self.conv2 = basic_block(out_channels,out_channels)
#         if in_channels != out_channels:
#             self.shortcut = False
#         else:
#             self.shortcut = True
#         if weight:
#             self.alpha =nn.Parameter(torch.ones(1))
#         else:
#             self.alpha = 1.0
#
#     def forward(self,x):
#         outputs = self.conv1(x)
#         outputs = self.conv2(outputs)
#         return outputs + self.alpha *x if self.shortcut else outputs
#
# class RepBlock(nn.Module):
#     '''RepBlock is a stage block with rep-style basic block'''
#
#     def init_(self, in_channels,out_channels,n=1,block=RepConv, basic_block=RepConv):
#         super().__init__()
#         self.conv1 = block(in_channels, out_channels)
#         self.block = nn.Sequential(*(block(out_channels, out_channels) for _ in range(n - 1))) if n>1 else None
#         if block == BottleRep:
#             self.conv1 = BottleRep(in_channels, out_channels, basic_block=basic_block, weight=True)
#             n = n // 2
#             self.block = nn.Sequential(*(BottleRep(out_channels,out_channels,basic_block=basic_block, weight=True) for _ in range(n - 1)))
#
#     def forward(self, x):
#         x= self.conv1(x)
#         if self.block is not None:
#             x= self.block(x)
#         return x
#
#
# class ADown(nn.Module):
#     """ADown."""
#
#     def __init__(self, c1, c2):
#         """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
#         super().__init__()
#         self.c = c2 // 2
#         self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
#         self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)
#
#     def forward(self, x):
#         """Forward pass through ADown layer."""
#         x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
#         x1, x2 = x.chunk(2, 1)
#         x1 = self.cv1(x1)
#         x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
#         x2 = self.cv2(x2)
#         return torch.cat((x1, x2), 1)
#
#
# class SPPELAN(nn.Module):
#     """SPP-ELAN."""
#
#     def __init__(self, c1, c2, c3, k=5):
#         """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
#         super().__init__()
#         self.c = c3
#         self.cv1 = Conv(c1, c3, 1, 1)
#         self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#         self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#         self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#         self.cv5 = Conv(4 * c3, c2, 1, 1)
#
#     def forward(self, x):
#         """Forward pass through SPPELAN layer."""
#         y = [self.cv1(x)]
#         y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
#         return self.cv5(torch.cat(y, 1))
#
#
# class Silence(nn.Module):
#     """Silence."""
#
#     def __init__(self):
#         """Initializes the Silence module."""
#         super(Silence, self).__init__()
#
#     def forward(self, x):
#         """Forward pass through Silence layer."""
#         return x
#
#
# class CBLinear(nn.Module):
#     """CBLinear."""
#
#     def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
#         """Initializes the CBLinear module, passing inputs unchanged."""
#         super(CBLinear, self).__init__()
#         self.c2s = c2s
#         self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)
#
#     def forward(self, x):
#         """Forward pass through CBLinear layer."""
#         outs = self.conv(x).split(self.c2s, dim=1)
#         return outs
#
#
# class CBFuse(nn.Module):
#     """CBFuse."""
#
#     def __init__(self, idx):
#         """Initializes CBFuse module with layer index for selective feature fusion."""
#         super(CBFuse, self).__init__()
#         self.idx = idx
#
#     def forward(self, xs):
#         """Forward pass through CBFuse layer."""
#         target_size = xs[-1].shape[2:]
#         res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
#         out = torch.sum(torch.stack(res + xs[-1:]), dim=0)
#         return out
#
# class TopkRouting(nn.Module):
#     """
#     differentiable topk routing with scaling
#     Args:
#         qk_dim: int, feature dimension of query and key
#         topk: int, the 'topk'
#         qk_scale: int or None, temperature (multiply) of softmax activation
#         with_param: bool, wether inorporate learnable params in routing unit
#         diff_routing: bool, wether make routing differentiable
#         soft_routing: bool, wether make output value multiplied by routing weights
#     """
#
#     def __init__(self, qk_dim, topk=4, qk_scale=None, param_routing=False, diff_routing=False):
#         super().__init__()
#         self.topk = topk
#         self.qk_dim = qk_dim
#         self.scale = qk_scale or qk_dim ** -0.5
#         self.diff_routing = diff_routing
#         # TODO: norm layer before/after linear?
#         self.emb = nn.Linear(qk_dim, qk_dim) if param_routing else nn.Identity()
#         # routing activation
#         self.routing_act = nn.Softmax(dim=-1)
#
#     def forward(self, query: Tensor, key: Tensor) -> Tuple[Tensor]:
#         """
#         Args:
#             q, k: (n, p^2, c) tensor
#         Return:
#             r_weight, topk_index: (n, p^2, topk) tensor
#         """
#         if not self.diff_routing:
#             query, key = query.detach(), key.detach()
#         query_hat, key_hat = self.emb(query), self.emb(key)  # per-window pooling -> (n, p^2, c)
#         attn_logit = (query_hat * self.scale) @ key_hat.transpose(-2, -1)  # (n, p^2, p^2)
#         topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)  # (n, p^2, k), (n, p^2, k)
#         r_weight = self.routing_act(topk_attn_logit)  # (n, p^2, k)
#
#         return r_weight, topk_index
#
#
# class KVGather(nn.Module):
#     def __init__(self, mul_weight='none'):
#         super().__init__()
#         assert mul_weight in ['none', 'soft', 'hard']
#         self.mul_weight = mul_weight
#
#     def forward(self, r_idx: Tensor, r_weight: Tensor, kv: Tensor):
#         """
#         r_idx: (n, p^2, topk) tensor
#         r_weight: (n, p^2, topk) tensor
#         kv: (n, p^2, w^2, c_kq+c_v)
#         Return:
#             (n, p^2, topk, w^2, c_kq+c_v) tensor
#         """
#         # select kv according to routing index
#         n, p2, w2, c_kv = kv.size()
#         topk = r_idx.size(-1)
#         # print(r_idx.size(), r_weight.size())
#         # FIXME: gather consumes much memory (topk times redundancy), write cuda kernel?
#         topk_kv = torch.gather(kv.view(n, 1, p2, w2, c_kv).expand(-1, p2, -1, -1, -1),
#                                # (n, p^2, p^2, w^2, c_kv) without mem cpy
#                                dim=2,
#                                index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_kv)
#                                # (n, p^2, k, w^2, c_kv)
#                                )
#
#         if self.mul_weight == 'soft':
#             topk_kv = r_weight.view(n, p2, topk, 1, 1) * topk_kv  # (n, p^2, k, w^2, c_kv)
#         elif self.mul_weight == 'hard':
#             raise NotImplementedError('differentiable hard routing TBA')
#         # else: #'none'
#         #     topk_kv = topk_kv # do nothing
#
#         return topk_kv
#
#
# class QKVLinear(nn.Module):
#     def __init__(self, dim, qk_dim, bias=True):
#         super().__init__()
#         self.dim = dim
#         self.qk_dim = qk_dim
#         self.qkv = nn.Linear(dim, qk_dim + qk_dim + dim, bias=bias)
#
#     def forward(self, x):
#         q, kv = self.qkv(x).split([self.qk_dim, self.qk_dim + self.dim], dim=-1)
#         return q, kv
#         # q, k, v = self.qkv(x).split([self.qk_dim, self.qk_dim, self.dim], dim=-1)
#         # return q, k, v
#
# class BiLevelRoutingAttention(nn.Module):
#     """
#     n_win: number of windows in one side (so the actual number of windows is n_win*n_win)
#     kv_per_win: for kv_downsample_mode='ada_xxxpool' only, number of key/values per window. Similar to n_win, the actual number is kv_per_win*kv_per_win.
#     topk: topk for window filtering
#     param_attention: 'qkvo'-linear for q,k,v and o, 'none': param free attention
#     param_routing: extra linear for routing
#     diff_routing: wether to set routing differentiable
#     soft_routing: wether to multiply soft routing weights
#     """
#
#     ###参数可自行修改
#     def __init__(self, dim, n_win=7, num_heads=8, qk_dim=None, qk_scale=None,
#                  kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='identity',
#                  topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False,
#                  side_dwconv=3,
#                  auto_pad=True):
#         super().__init__()
#         # local attention setting
#         self.dim = dim
#         self.n_win = n_win  # Wh, Ww
#         self.num_heads = num_heads
#         self.qk_dim = qk_dim or dim
#         assert self.qk_dim % num_heads == 0 and self.dim % num_heads == 0, 'qk_dim and dim must be divisible by num_heads!'
#         self.scale = qk_scale or self.qk_dim ** -0.5
#
#         ################side_dwconv (i.e. LCE in ShuntedTransformer)###########
#         self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv // 2,
#                               groups=dim) if side_dwconv > 0 else \
#             lambda x: torch.zeros_like(x)
#
#         ################ global routing setting #################
#         self.topk = topk
#         self.param_routing = param_routing
#         self.diff_routing = diff_routing
#         self.soft_routing = soft_routing
#         # router
#         assert not (self.param_routing and not self.diff_routing)  # cannot be with_param=True and diff_routing=False
#         self.router = TopkRouting(qk_dim=self.qk_dim,
#                                   qk_scale=self.scale,
#                                   topk=self.topk,
#                                   diff_routing=self.diff_routing,
#                                   param_routing=self.param_routing)
#         if self.soft_routing:  # soft routing, always diffrentiable (if no detach)
#             mul_weight = 'soft'
#         elif self.diff_routing:  # hard differentiable routing
#             mul_weight = 'hard'
#         else:  # hard non-differentiable routing
#             mul_weight = 'none'
#         self.kv_gather = KVGather(mul_weight=mul_weight)
#
#         # qkv mapping (shared by both global routing and local attention)
#         self.param_attention = param_attention
#         if self.param_attention == 'qkvo':
#             self.qkv = QKVLinear(self.dim, self.qk_dim)
#             self.wo = nn.Linear(dim, dim)
#         elif self.param_attention == 'qkv':
#             self.qkv = QKVLinear(self.dim, self.qk_dim)
#             self.wo = nn.Identity()
#         else:
#             raise ValueError(f'param_attention mode {self.param_attention} is not surpported!')
#
#         self.kv_downsample_mode = kv_downsample_mode
#         self.kv_per_win = kv_per_win
#         self.kv_downsample_ratio = kv_downsample_ratio
#         self.kv_downsample_kenel = kv_downsample_kernel
#         if self.kv_downsample_mode == 'ada_avgpool':
#             assert self.kv_per_win is not None
#             self.kv_down = nn.AdaptiveAvgPool2d(self.kv_per_win)
#         elif self.kv_downsample_mode == 'ada_maxpool':
#             assert self.kv_per_win is not None
#             self.kv_down = nn.AdaptiveMaxPool2d(self.kv_per_win)
#         elif self.kv_downsample_mode == 'maxpool':
#             assert self.kv_downsample_ratio is not None
#             self.kv_down = nn.MaxPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
#         elif self.kv_downsample_mode == 'avgpool':
#             assert self.kv_downsample_ratio is not None
#             self.kv_down = nn.AvgPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
#         elif self.kv_downsample_mode == 'identity':  # no kv downsampling
#             self.kv_down = nn.Identity()
#         elif self.kv_downsample_mode == 'fracpool':
#             # assert self.kv_downsample_ratio is not None
#             # assert self.kv_downsample_kenel is not None
#             # TODO: fracpool
#             # 1. kernel size should be input size dependent
#             # 2. there is a random factor, need to avoid independent sampling for k and v
#             raise NotImplementedError('fracpool policy is not implemented yet!')
#         elif kv_downsample_mode == 'conv':
#             # TODO: need to consider the case where k != v so that need two downsample modules
#             raise NotImplementedError('conv policy is not implemented yet!')
#         else:
#             raise ValueError(f'kv_down_sample_mode {self.kv_downsaple_mode} is not surpported!')
#
#         # softmax for local attention
#         self.attn_act = nn.Softmax(dim=-1)
#
#         self.auto_pad = auto_pad
#
#     def forward(self, x, ret_attn_mask=False):
#         """
#         x: NHWC tensor
#         Return:
#             NHWC tensor
#         """
#         x = rearrange(x, "n c h w -> n h w c")
#         # NOTE: use padding for semantic segmentation
#         ###################################################
#         if self.auto_pad:
#             N, H_in, W_in, C = x.size()
#
#             pad_l = pad_t = 0
#             pad_r = (self.n_win - W_in % self.n_win) % self.n_win
#             pad_b = (self.n_win - H_in % self.n_win) % self.n_win
#             x = F.pad(x, (0, 0,  # dim=-1
#                           pad_l, pad_r,  # dim=-2
#                           pad_t, pad_b))  # dim=-3
#             _, H, W, _ = x.size()  # padded size
#         else:
#             N, H, W, C = x.size()
#             assert H % self.n_win == 0 and W % self.n_win == 0  #
#         ###################################################
#
#         # patchify, (n, p^2, w, w, c), keep 2d window as we need 2d pooling to reduce kv size
#         x = rearrange(x, "n (j h) (i w) c -> n (j i) h w c", j=self.n_win, i=self.n_win)
#
#         #################qkv projection###################
#         # q: (n, p^2, w, w, c_qk)
#         # kv: (n, p^2, w, w, c_qk+c_v)
#         # NOTE: separte kv if there were memory leak issue caused by gather
#         q, kv = self.qkv(x)
#
#         # pixel-wise qkv
#         # q_pix: (n, p^2, w^2, c_qk)
#         # kv_pix: (n, p^2, h_kv*w_kv, c_qk+c_v)
#         q_pix = rearrange(q, 'n p2 h w c -> n p2 (h w) c')
#         kv_pix = self.kv_down(rearrange(kv, 'n p2 h w c -> (n p2) c h w'))
#         kv_pix = rearrange(kv_pix, '(n j i) c h w -> n (j i) (h w) c', j=self.n_win, i=self.n_win)
#
#         q_win, k_win = q.mean([2, 3]), kv[..., 0:self.qk_dim].mean(
#             [2, 3])  # window-wise qk, (n, p^2, c_qk), (n, p^2, c_qk)
#
#         ##################side_dwconv(lepe)##################
#         # NOTE: call contiguous to avoid gradient warning when using ddp
#         lepe = self.lepe(rearrange(kv[..., self.qk_dim:], 'n (j i) h w c -> n c (j h) (i w)', j=self.n_win,
#                                    i=self.n_win).contiguous())
#         lepe = rearrange(lepe, 'n c (j h) (i w) -> n (j h) (i w) c', j=self.n_win, i=self.n_win)
#
#         ############ gather q dependent k/v #################
#
#         r_weight, r_idx = self.router(q_win, k_win)  # both are (n, p^2, topk) tensors
#
#         kv_pix_sel = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kv_pix)  # (n, p^2, topk, h_kv*w_kv, c_qk+c_v)
#         k_pix_sel, v_pix_sel = kv_pix_sel.split([self.qk_dim, self.dim], dim=-1)
#         # kv_pix_sel: (n, p^2, topk, h_kv*w_kv, c_qk)
#         # v_pix_sel: (n, p^2, topk, h_kv*w_kv, c_v)
#
#         ######### do attention as normal ####################
#         k_pix_sel = rearrange(k_pix_sel, 'n p2 k w2 (m c) -> (n p2) m c (k w2)',
#                               m=self.num_heads)  # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_kq//m) transpose here?
#         v_pix_sel = rearrange(v_pix_sel, 'n p2 k w2 (m c) -> (n p2) m (k w2) c',
#                               m=self.num_heads)  # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_v//m)
#         q_pix = rearrange(q_pix, 'n p2 w2 (m c) -> (n p2) m w2 c',
#                           m=self.num_heads)  # to BMLC tensor (n*p^2, m, w^2, c_qk//m)
#
#         # param-free multihead attention
#         attn_weight = (
#                                   q_pix * self.scale) @ k_pix_sel  # (n*p^2, m, w^2, c) @ (n*p^2, m, c, topk*h_kv*w_kv) -> (n*p^2, m, w^2, topk*h_kv*w_kv)
#         attn_weight = self.attn_act(attn_weight)
#         out = attn_weight @ v_pix_sel  # (n*p^2, m, w^2, topk*h_kv*w_kv) @ (n*p^2, m, topk*h_kv*w_kv, c) -> (n*p^2, m, w^2, c)
#         out = rearrange(out, '(n j i) m (h w) c -> n (j h) (i w) (m c)', j=self.n_win, i=self.n_win,
#                         h=H // self.n_win, w=W // self.n_win)
#
#         out = out + lepe
#         # output linear
#         out = self.wo(out)
#
#         # NOTE: use padding for semantic segmentation
#         # crop padded region
#         if self.auto_pad and (pad_r > 0 or pad_b > 0):
#             out = out[:, :H_in, :W_in, :].contiguous()
#
#         if ret_attn_mask:
#             return out, r_weight, r_idx, attn_weight
#         else:
#             return rearrange(out, "n h w c -> n c h w")
#
# class BiLevelRoutingAttention(nn.Module):
#     """
#     n_win: number of windows in one side (so the actual number of windows is n_win*n_win)
#     kv_per_win: for kv_downsample_mode='ada_xxxpool' only, number of key/values per window. Similar to n_win, the actual number is kv_per_win*kv_per_win.
#     topk: topk for window filtering
#     param_attention: 'qkvo'-linear for q,k,v and o, 'none': param free attention
#     param_routing: extra linear for routing
#     diff_routing: wether to set routing differentiable
#     soft_routing: wether to multiply soft routing weights
#     """
#
#     ###参数可自行修改
#     def __init__(self, dim, n_win=7, num_heads=8, qk_dim=None, qk_scale=None,
#                  kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='identity',
#                  topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False,
#                  side_dwconv=3,
#                  auto_pad=True):
#         super().__init__()
#         # local attention setting
#         self.dim = dim
#         self.n_win = n_win  # Wh, Ww
#         self.num_heads = num_heads
#         self.qk_dim = qk_dim or dim
#         assert self.qk_dim % num_heads == 0 and self.dim % num_heads == 0, 'qk_dim and dim must be divisible by num_heads!'
#         self.scale = qk_scale or self.qk_dim ** -0.5
#
#         ################side_dwconv (i.e. LCE in ShuntedTransformer)###########
#         self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv // 2,
#                               groups=dim) if side_dwconv > 0 else \
#             lambda x: torch.zeros_like(x)
#
#         ################ global routing setting #################
#         self.topk = topk
#         self.param_routing = param_routing
#         self.diff_routing = diff_routing
#         self.soft_routing = soft_routing
#         # router
#         assert not (self.param_routing and not self.diff_routing)  # cannot be with_param=True and diff_routing=False
#         self.router = TopkRouting(qk_dim=self.qk_dim,
#                                   qk_scale=self.scale,
#                                   topk=self.topk,
#                                   diff_routing=self.diff_routing,
#                                   param_routing=self.param_routing)
#         if self.soft_routing:  # soft routing, always diffrentiable (if no detach)
#             mul_weight = 'soft'
#         elif self.diff_routing:  # hard differentiable routing
#             mul_weight = 'hard'
#         else:  # hard non-differentiable routing
#             mul_weight = 'none'
#         self.kv_gather = KVGather(mul_weight=mul_weight)
#
#         # qkv mapping (shared by both global routing and local attention)
#         self.param_attention = param_attention
#         if self.param_attention == 'qkvo':
#             self.qkv = QKVLinear(self.dim, self.qk_dim)
#             self.wo = nn.Linear(dim, dim)
#         elif self.param_attention == 'qkv':
#             self.qkv = QKVLinear(self.dim, self.qk_dim)
#             self.wo = nn.Identity()
#         else:
#             raise ValueError(f'param_attention mode {self.param_attention} is not surpported!')
#
#         self.kv_downsample_mode = kv_downsample_mode
#         self.kv_per_win = kv_per_win
#         self.kv_downsample_ratio = kv_downsample_ratio
#         self.kv_downsample_kenel = kv_downsample_kernel
#         if self.kv_downsample_mode == 'ada_avgpool':
#             assert self.kv_per_win is not None
#             self.kv_down = nn.AdaptiveAvgPool2d(self.kv_per_win)
#         elif self.kv_downsample_mode == 'ada_maxpool':
#             assert self.kv_per_win is not None
#             self.kv_down = nn.AdaptiveMaxPool2d(self.kv_per_win)
#         elif self.kv_downsample_mode == 'maxpool':
#             assert self.kv_downsample_ratio is not None
#             self.kv_down = nn.MaxPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
#         elif self.kv_downsample_mode == 'avgpool':
#             assert self.kv_downsample_ratio is not None
#             self.kv_down = nn.AvgPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
#         elif self.kv_downsample_mode == 'identity':  # no kv downsampling
#             self.kv_down = nn.Identity()
#         elif self.kv_downsample_mode == 'fracpool':
#             # assert self.kv_downsample_ratio is not None
#             # assert self.kv_downsample_kenel is not None
#             # TODO: fracpool
#             # 1. kernel size should be input size dependent
#             # 2. there is a random factor, need to avoid independent sampling for k and v
#             raise NotImplementedError('fracpool policy is not implemented yet!')
#         elif kv_downsample_mode == 'conv':
#             # TODO: need to consider the case where k != v so that need two downsample modules
#             raise NotImplementedError('conv policy is not implemented yet!')
#         else:
#             raise ValueError(f'kv_down_sample_mode {self.kv_downsaple_mode} is not surpported!')
#
#         # softmax for local attention
#         self.attn_act = nn.Softmax(dim=-1)
#
#         self.auto_pad = auto_pad
#
#     def forward(self, x, ret_attn_mask=False):
#         """
#         x: NHWC tensor
#         Return:
#             NHWC tensor
#         """
#         x = rearrange(x, "n c h w -> n h w c")
#         # NOTE: use padding for semantic segmentation
#         ###################################################
#         if self.auto_pad:
#             N, H_in, W_in, C = x.size()
#
#             pad_l = pad_t = 0
#             pad_r = (self.n_win - W_in % self.n_win) % self.n_win
#             pad_b = (self.n_win - H_in % self.n_win) % self.n_win
#             x = F.pad(x, (0, 0,  # dim=-1
#                           pad_l, pad_r,  # dim=-2
#                           pad_t, pad_b))  # dim=-3
#             _, H, W, _ = x.size()  # padded size
#         else:
#             N, H, W, C = x.size()
#             assert H % self.n_win == 0 and W % self.n_win == 0  #
#         ###################################################
#
#         # patchify, (n, p^2, w, w, c), keep 2d window as we need 2d pooling to reduce kv size
#         x = rearrange(x, "n (j h) (i w) c -> n (j i) h w c", j=self.n_win, i=self.n_win)
#
#         #################qkv projection###################
#         # q: (n, p^2, w, w, c_qk)
#         # kv: (n, p^2, w, w, c_qk+c_v)
#         # NOTE: separte kv if there were memory leak issue caused by gather
#         q, kv = self.qkv(x)
#
#         # pixel-wise qkv
#         # q_pix: (n, p^2, w^2, c_qk)
#         # kv_pix: (n, p^2, h_kv*w_kv, c_qk+c_v)
#         q_pix = rearrange(q, 'n p2 h w c -> n p2 (h w) c')
#         kv_pix = self.kv_down(rearrange(kv, 'n p2 h w c -> (n p2) c h w'))
#         kv_pix = rearrange(kv_pix, '(n j i) c h w -> n (j i) (h w) c', j=self.n_win, i=self.n_win)
#
#         q_win, k_win = q.mean([2, 3]), kv[..., 0:self.qk_dim].mean(
#             [2, 3])  # window-wise qk, (n, p^2, c_qk), (n, p^2, c_qk)
#
#         ##################side_dwconv(lepe)##################
#         # NOTE: call contiguous to avoid gradient warning when using ddp
#         lepe = self.lepe(rearrange(kv[..., self.qk_dim:], 'n (j i) h w c -> n c (j h) (i w)', j=self.n_win,
#                                    i=self.n_win).contiguous())
#         lepe = rearrange(lepe, 'n c (j h) (i w) -> n (j h) (i w) c', j=self.n_win, i=self.n_win)
#
#         ############ gather q dependent k/v #################
#
#         r_weight, r_idx = self.router(q_win, k_win)  # both are (n, p^2, topk) tensors
#
#         kv_pix_sel = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kv_pix)  # (n, p^2, topk, h_kv*w_kv, c_qk+c_v)
#         k_pix_sel, v_pix_sel = kv_pix_sel.split([self.qk_dim, self.dim], dim=-1)
#         # kv_pix_sel: (n, p^2, topk, h_kv*w_kv, c_qk)
#         # v_pix_sel: (n, p^2, topk, h_kv*w_kv, c_v)
#
#         ######### do attention as normal ####################
#         k_pix_sel = rearrange(k_pix_sel, 'n p2 k w2 (m c) -> (n p2) m c (k w2)',
#                               m=self.num_heads)  # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_kq//m) transpose here?
#         v_pix_sel = rearrange(v_pix_sel, 'n p2 k w2 (m c) -> (n p2) m (k w2) c',
#                               m=self.num_heads)  # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_v//m)
#         q_pix = rearrange(q_pix, 'n p2 w2 (m c) -> (n p2) m w2 c',
#                           m=self.num_heads)  # to BMLC tensor (n*p^2, m, w^2, c_qk//m)
#
#         # param-free multihead attention
#         attn_weight = (
#                                   q_pix * self.scale) @ k_pix_sel  # (n*p^2, m, w^2, c) @ (n*p^2, m, c, topk*h_kv*w_kv) -> (n*p^2, m, w^2, topk*h_kv*w_kv)
#         attn_weight = self.attn_act(attn_weight)
#         out = attn_weight @ v_pix_sel  # (n*p^2, m, w^2, topk*h_kv*w_kv) @ (n*p^2, m, topk*h_kv*w_kv, c) -> (n*p^2, m, w^2, c)
#         out = rearrange(out, '(n j i) m (h w) c -> n (j h) (i w) (m c)', j=self.n_win, i=self.n_win,
#                         h=H // self.n_win, w=W // self.n_win)
#
#         out = out + lepe
#         # output linear
#         out = self.wo(out)
#
#         # NOTE: use padding for semantic segmentation
#         # crop padded region
#         if self.auto_pad and (pad_r > 0 or pad_b > 0):
#             out = out[:, :H_in, :W_in, :].contiguous()
#
#         if ret_attn_mask:
#             return out, r_weight, r_idx, attn_weight
#         else:
#             return rearrange(out, "n h w c -> n c h w")
# class Add(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x):
#         return torch.sum(torch.stack(x, dim=0), dim=0)
#
# class Zoom_cat(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x):
#         l, m, s = x[0], x[1], x[2]
#         tgt_size = m.shape[2:]
#         l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
#         s = F.interpolate(s, m.shape[2:], mode='nearest')
#         lms = torch.cat([l, m, s], dim=1)
#         return lms
#
# class ScalSeq(nn.Module):
#     def __init__(self, inc, channel):
#         super(ScalSeq, self).__init__()
#         if channel != inc[0]:
#             self.conv0 = Conv(inc[0], channel,1)
#         self.conv1 =  Conv(inc[1], channel,1)
#         self.conv2 =  Conv(inc[2], channel,1)
#         self.conv3d = nn.Conv3d(channel,channel,kernel_size=(1,1,1))
#         self.bn = nn.BatchNorm3d(channel)
#         self.act = nn.LeakyReLU(0.1)
#         self.pool_3d = nn.MaxPool3d(kernel_size=(3,1,1))
#
#     def forward(self, x):
#         p3, p4, p5 = x[0],x[1],x[2]
#         if hasattr(self, 'conv0'):
#             p3 = self.conv0(p3)
#         p4_2 = self.conv1(p4)
#         p4_2 = F.interpolate(p4_2, p3.size()[2:], mode='nearest')
#         p5_2 = self.conv2(p5)
#         p5_2 = F.interpolate(p5_2, p3.size()[2:], mode='nearest')
#         p3_3d = torch.unsqueeze(p3, -3)
#         p4_3d = torch.unsqueeze(p4_2, -3)
#         p5_3d = torch.unsqueeze(p5_2, -3)
#         combine = torch.cat([p3_3d, p4_3d, p5_3d],dim = 2)
#         conv_3d = self.conv3d(combine)
#         bn = self.bn(conv_3d)
#         act = self.act(bn)
#         x = self.pool_3d(act)
#         x = torch.squeeze(x, 2)
#         return x
#
#
# class Faster_Block_EMA(nn.Module):
#     def __init__(self,
#                  inc,
#                  dim,
#                  n_div=4,
#                  mlp_ratio=2,
#                  drop_path=0.1,
#                  layer_scale_init_value=0.0,
#                  pconv_fw_type='split_cat'
#                  ):
#         super().__init__()
#         self.dim = dim
#         self.mlp_ratio = mlp_ratio
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.n_div = n_div
#
#         mlp_hidden_dim = int(dim * mlp_ratio)
#
#         mlp_layer = [
#             Conv(dim, mlp_hidden_dim, 1),
#             nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
#         ]
#
#         self.mlp = nn.Sequential(*mlp_layer)
#
#         self.spatial_mixing = Partial_conv3(
#             dim,
#             n_div,
#             pconv_fw_type
#         )
#         self.attention = EMA(dim)
#
#         self.adjust_channel = None
#         if inc != dim:
#             self.adjust_channel = Conv(inc, dim, 1)
#
#         if layer_scale_init_value > 0:
#             self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
#             self.forward = self.forward_layer_scale
#         else:
#             self.forward = self.forward
#
#     def forward(self, x):
#         if self.adjust_channel is not None:
#             x = self.adjust_channel(x)
#         shortcut = x
#         x = self.spatial_mixing(x)
#         x = shortcut + self.attention(self.drop_path(self.mlp(x)))
#         return x
#
#     def forward_layer_scale(self, x):
#         shortcut = x
#         x = self.spatial_mixing(x)
#         x = shortcut + self.drop_path(self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
#         return x
# class C2f_Faster_EMA(C2f):
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
#         super().__init__(c1, c2, n, shortcut, g, e)
#         self.m = nn.ModuleList(Faster_Block_EMA(self.c, self.c) for _ in range(n))
#
#
#
# class Bottleneck_DBB(Bottleneck):
#     def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
#         super().__init__(c1, c2, shortcut, g, k, e)
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = DiverseBranchBlock(c1, c_, k[0], 1)
#         self.cv2 = DiverseBranchBlock(c_, c2, k[1], 1, groups=g)
# class C2f_DBB(C2f):
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
#         super().__init__(c1, c2, n, shortcut, g, e)
#         self.m = nn.ModuleList(Bottleneck_DBB(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))
#
# # gfpn
# class CSPStage(nn.Module):
#     def __init__(self,
#                  ch_in,
#                  ch_out,
#                  n,
#                  block_fn='BasicBlock_3x3_Reverse',
#                  ch_hidden_ratio=1.0,
#                  act='silu',
#                  spp=False):
#         super(CSPStage, self).__init__()
#
#         split_ratio = 2
#         ch_first = int(ch_out // split_ratio)
#         ch_mid = int(ch_out - ch_first)
#         self.conv1 = Conv(ch_in, ch_first, 1)
#         self.conv2 = Conv(ch_in, ch_mid, 1)
#         self.convs = nn.Sequential()
#
#         next_ch_in = ch_mid
#         for i in range(n):
#             if block_fn == 'BasicBlock_3x3_Reverse':
#                 self.convs.add_module(
#                     str(i),
#                     BasicBlock_3x3_Reverse(next_ch_in,
#                                            ch_hidden_ratio,
#                                            ch_mid,
#                                            shortcut=True))
#             else:
#                 raise NotImplementedError
#             if i == (n - 1) // 2 and spp:
#                 self.convs.add_module('spp', SPP(ch_mid * 4, ch_mid, 1, [5, 9, 13]))
#             next_ch_in = ch_mid
#         self.conv3 = Conv(ch_mid * n + ch_first, ch_out, 1)
#
#     def forward(self, x):
#         y1 = self.conv1(x)
#         y2 = self.conv2(x)
#
#         mid_out = [y1]
#         for conv in self.convs:
#             y2 = conv(y2)
#             mid_out.append(y2)
#         y = torch.cat(mid_out, axis=1)
#         y = self.conv3(y)
#         return y
#
# class BasicBlock_3x3_Reverse(nn.Module):
#     def __init__(self,
#                  ch_in,
#                  ch_hidden_ratio,
#                  ch_out,
#                  shortcut=True):
#         super(BasicBlock_3x3_Reverse, self).__init__()
#         assert ch_in == ch_out
#         ch_hidden = int(ch_in * ch_hidden_ratio)
#         self.conv1 = Conv(ch_hidden, ch_out, 3, s=1)
#         self.conv2 = RepConv(ch_in, ch_hidden, 3, s=1)
#         self.shortcut = shortcut
#
#     def forward(self, x):
#         y = self.conv2(x)
#         y = self.conv1(y)
#         if self.shortcut:
#             return x + y
#         else:
#             return y
#
#
#
# class GSConv(nn.Module):
#     # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
#         super().__init__()
#         c_ = c2 // 2
#         self.cv1 = Conv(c1, c_, k, s, p, g, d, Conv.default_act)
#         self.cv2 = Conv(c_, c_, 5, 1, p, c_, d, Conv.default_act)
#
#     def forward(self, x):
#         x1 = self.cv1(x)
#         x2 = torch.cat((x1, self.cv2(x1)), 1)
#         # shuffle
#         # y = x2.reshape(x2.shape[0], 2, x2.shape[1] // 2, x2.shape[2], x2.shape[3])
#         # y = y.permute(0, 2, 1, 3, 4)
#         # return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])
#
#         b, n, h, w = x2.size()
#         b_n = b * n // 2
#         y = x2.reshape(b_n, 2, h * w)
#         y = y.permute(1, 0, 2)
#         y = y.reshape(2, -1, n // 2, h, w)
#
#         return torch.cat((y[0], y[1]), 1)
#
#
# class GSConvns(GSConv):
#     # GSConv with a normative-shuffle https://github.com/AlanLi1997/slim-neck-by-gsconv
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
#         super().__init__(c1, c2, k, s, p, g, act=True)
#         c_ = c2 // 2
#         self.shuf = nn.Conv2d(c_ * 2, c2, 1, 1, 0, bias=False)
#
#     def forward(self, x):
#         x1 = self.cv1(x)
#         x2 = torch.cat((x1, self.cv2(x1)), 1)
#         # normative-shuffle, TRT supported
#         return nn.ReLU()(self.shuf(x2))
#
#
# class GSBottleneck(nn.Module):
#     # GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
#     def __init__(self, c1, c2, k=3, s=1, e=0.5):
#         super().__init__()
#         c_ = int(c2 * e)
#         # for lighting
#         self.conv_lighting = nn.Sequential(
#             GSConv(c1, c_, 1, 1),
#             GSConv(c_, c2, 3, 1, act=False))
#         self.shortcut = Conv(c1, c2, 1, 1, act=False)
#
#     def forward(self, x):
#         return self.conv_lighting(x) + self.shortcut(x)
#
#
# class GSBottleneckns(GSBottleneck):
#     # GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
#     def __init__(self, c1, c2, k=3, s=1, e=0.5):
#         super().__init__(c1, c2, k, s, e)
#         c_ = int(c2 * e)
#         # for lighting
#         self.conv_lighting = nn.Sequential(
#             GSConvns(c1, c_, 1, 1),
#             GSConvns(c_, c2, 3, 1, act=False))
#
#
# class GSBottleneckC(GSBottleneck):
#     # cheap GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
#     def __init__(self, c1, c2, k=3, s=1):
#         super().__init__(c1, c2, k, s)
#         self.shortcut = DWConv(c1, c2, k, s, act=False)
#
#
# class VoVGSCSP(nn.Module):
#     # VoVGSCSP module with GSBottleneck
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c1, c_, 1, 1)
#         self.gsb = nn.Sequential(*(GSBottleneck(c_, c_, e=1.0) for _ in range(n)))
#         self.res = Conv(c_, c_, 3, 1, act=False)
#         self.cv3 = Conv(2 * c_, c2, 1)
#
#     def forward(self, x):
#         x1 = self.gsb(self.cv1(x))
#         y = self.cv2(x)
#         return self.cv3(torch.cat((y, x1), dim=1))
#
#
# class VoVGSCSPns(VoVGSCSP):
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         super().__init__(c1, c2, n, shortcut, g, e)
#         c_ = int(c2 * e)  # hidden channels
#         self.gsb = nn.Sequential(*(GSBottleneckns(c_, c_, e=1.0) for _ in range(n)))
#
#
# class VoVGSCSPC(VoVGSCSP):
#     # cheap VoVGSCSP module with GSBottleneck
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         super().__init__(c1, c2)
#         c_ = int(c2 * 0.5)  # hidden channels
#         self.gsb = GSBottleneckC(c_, c_, 1, 1)
#
# import torch.nn.functional as F
# class SDI(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#
#         # self.convs = nn.ModuleList([nn.Conv2d(channel, channels[0], kernel_size=3, stride=1, padding=1) for channel in channels])
#         self.convs = nn.ModuleList([GSConv(channel, channels[0]) for channel in channels])
#
#     def forward(self, xs):
#         ans = torch.ones_like(xs[0])
#         target_size = xs[0].shape[2:]
#         for i, x in enumerate(xs):
#             if x.shape[-1] > target_size[-1]:
#                 x = F.adaptive_avg_pool2d(x, (target_size[0], target_size[1]))
#             elif x.shape[-1] < target_size[-1]:
#                 x = F.interpolate(x, size=(target_size[0], target_size[1]),
#                                       mode='bilinear', align_corners=True)
#             ans = ans * self.convs[i](x)
#         return ans
#
#
# class Fusion(nn.Module):
#     def __init__(self, inc_list, fusion='bifpn') -> None:
#         super().__init__()
#
#         assert fusion in ['weight', 'adaptive', 'concat', 'bifpn', 'SDI']
#         self.fusion = fusion
#
#         if self.fusion == 'bifpn':
#             self.fusion_weight = nn.Parameter(torch.ones(len(inc_list), dtype=torch.float32), requires_grad=True)
#             self.relu = nn.ReLU()
#             self.epsilon = 1e-4
#         elif self.fusion == 'SDI':
#             self.SDI = SDI(inc_list)
#         else:
#             self.fusion_conv = nn.ModuleList([Conv(inc, inc, 1) for inc in inc_list])
#
#             if self.fusion == 'adaptive':
#                 self.fusion_adaptive = Conv(sum(inc_list), len(inc_list), 1)
#
#     def forward(self, x):
#         if self.fusion in ['weight', 'adaptive']:
#             for i in range(len(x)):
#                 x[i] = self.fusion_conv[i](x[i])
#         if self.fusion == 'weight':
#             return torch.sum(torch.stack(x, dim=0), dim=0)
#         elif self.fusion == 'adaptive':
#             fusion = torch.softmax(self.fusion_adaptive(torch.cat(x, dim=1)), dim=1)
#             x_weight = torch.split(fusion, [1] * len(x), dim=1)
#             return torch.sum(torch.stack([x_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)
#         elif self.fusion == 'concat':
#             return torch.cat(x, dim=1)
#         elif self.fusion == 'bifpn':
#             fusion_weight = self.relu(self.fusion_weight.clone())
#             fusion_weight = fusion_weight / (torch.sum(fusion_weight, dim=0))
#             return torch.sum(torch.stack([fusion_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)
#         elif self.fusion == 'SDI':
#             return self.SDI(x)
#
# ####################################### Focus Diffusion Pyramid Network end ########################################
#
# class FocusFeature(nn.Module):
#     def __init__(self, inc, kernel_sizes=(5, 7, 9, 11), e=0.5) -> None:
#         super().__init__()
#         hidc = int(inc[1] * e)
#
#         self.conv1 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             Conv(inc[0], hidc, 1)
#         )
#         self.conv2 = Conv(inc[1], hidc, 1) if e != 1 else nn.Identity()
#         self.conv3 = ADown(inc[2], hidc)
#
#         self.dw_conv = nn.ModuleList(
#             nn.Conv2d(hidc * 3, hidc * 3, kernel_size=k, padding=autopad(k), groups=hidc * 3) for k in kernel_sizes)
#         self.pw_conv = Conv(hidc * 3, hidc * 3)
#
#     def forward(self, x):
#         x1, x2, x3 = x
#         x1 = self.conv1(x1)
#         x2 = self.conv2(x2)
#         x3 = self.conv3(x3)
#
#         x = torch.cat([x1, x2, x3], dim=1)
#         feature = torch.sum(torch.stack([x] + [layer(x) for layer in self.dw_conv], dim=0), dim=0)
#         feature = self.pw_conv(feature)
#
#         x = x + feature
#         return x
# try:
#     from mmcv.cnn import build_activation_layer, build_norm_layer
#     from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d
#     from mmengine.model import constant_init, normal_init
# except ImportError as e:
#     pass
#
# class DyDCNv2(nn.Module):
#     """ModulatedDeformConv2d with normalization layer used in DyHead.
#     This module cannot be configured with `conv_cfg=dict(type='DCNv2')`
#     because DyHead calculates offset and mask from middle-level feature.
#     Args:
#         in_channels (int): Number of input channels.
#         out_channels (int): Number of output channels.
#         stride (int | tuple[int], optional): Stride of the convolution.
#             Default: 1.
#         norm_cfg (dict, optional): Config dict for normalization layer.
#             Default: dict(type='GN', num_groups=16, requires_grad=True).
#     """
#
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  stride=1,
#                  norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)):
#         super().__init__()
#         self.with_norm = norm_cfg is not None
#         bias = not self.with_norm
#         self.conv = ModulatedDeformConv2d(
#             in_channels, out_channels, 3, stride=stride, padding=1, bias=bias)
#         if self.with_norm:
#             self.norm = build_norm_layer(norm_cfg, out_channels)[1]
#
#     def forward(self, x, offset, mask):
#         """Forward function."""
#         x = self.conv(x.contiguous(), offset, mask)
#         if self.with_norm:
#             x = self.norm(x)
#         return x


# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""
from typing import Tuple

from einops import rearrange
from torch import Tensor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rep_block import *
from .attention import *
from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock
from ultralytics.utils.torch_utils import make_divisible
from ..backbone.repvit import Conv2d_BN, RepVGGDW, SqueezeExcite
__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ADown",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "Silence",
    "BiLevelRoutingAttention",
    'RepBlock',
    'ScalSeq',
    'Add',
    'Zoom_cat',
    'C2f_Faster',
    'C2f_Faster_EMA',
    'C2f_DBB',
    'CSPStage',
    'Fusion',
    'FocusFeature',
    'C2f_PKIModule',
    'CGAFusion',
    'CSMHSA',
    'C2f_RVB',
    'C3_RVB_SE',
    'C2f_RVB_SE',
    'C3_RVB_EMA',
    'C2f_RVB_EMA',


)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck with 2 convolutions module with arguments ch_in, ch_out, number, shortcut,
        groups, expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Contrastive Head for YOLO-World compute the region-text scores according to the similarity between image and text
    features.
    """

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcut option, groups and expansion
        ratio.
        """
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Rep CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

'''添加代码'''
# c2f
from timm.models.layers import DropPath


class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


class Faster_Block(nn.Module):
    def __init__(self,
                 inc,
                 dim,
                 n_div=4,
                 mlp_ratio=2,
                 drop_path=0.1,
                 layer_scale_init_value=0.0,
                 pconv_fw_type='split_cat'
                 ):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer = [
            Conv(dim, mlp_hidden_dim, 1),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )

        self.adjust_channel = None
        if inc != dim:
            self.adjust_channel = Conv(inc, dim, 1)

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x):
        if self.adjust_channel is not None:
            x = self.adjust_channel(x)
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class C3_Faster(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Faster_Block(c_, c_) for _ in range(n)))


class C2f_Faster(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Faster_Block(self.c, self.c) for _ in range(n))


# repblock
def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result
class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if use_se:
            #   Note that RepVGG-D2se uses SE before nonlinearity. But RepVGGplus models uses SE after nonlinearity.
            # self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
            raise  NotImplementedError('se block not supported')
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)
            print('RepVGG Block, identity = ', self.rbr_identity)


    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))


    #   Optional. This may improve the accuracy and facilitates quantization in some cases.
    #   1.  Cancel the original weight decay on rbr_dense.conv.weight and rbr_1x1.conv.weight.
    #   2.  Use like this.
    #       loss = criterion(....)
    #       for every RepVGGBlock blk:
    #           loss += weight_decay_coefficient * 0.5 * blk.get_cust_L2()
    #       optimizer.zero_grad()
    #       loss.backward()
    def get_custom_L2(self):
        K3 = self.rbr_dense.conv.weight
        K1 = self.rbr_1x1.conv.weight
        t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
        t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()

        l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()      # The L2 loss of the "circle" of weights in 3x3 kernel. Use regular L2 on them.
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1                           # The equivalent resultant central point of 3x3 kernel.
        l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()        # Normalize for an L2 coefficient comparable to regular L2.
        return l2_loss_eq_kernel + l2_loss_circle



#   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
#   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
#   May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1,1,1,1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True

class BottleRep(nn.Module):
    def __init__(self, in_channels, out_channels, basic_block=RepVGGBlock, weight=False):
        super().__init__()
        self.conv1 = basic_block(in_channels,out_channels)
        self.conv2 = basic_block(out_channels,out_channels)
        if in_channels != out_channels:
            self.shortcut = False
        else:
            self.shortcut = True
        if weight:
            self.alpha =nn.Parameter(torch.ones(1))
        else:
            self.alpha = 1.0

    def forward(self,x):
        outputs = self.conv1(x)
        outputs = self.conv2(outputs)
        return outputs + self.alpha *x if self.shortcut else outputs

class RepBlock(nn.Module):
    '''RepBlock is a stage block with rep-style basic block'''

    def init_(self, in_channels,out_channels,n=1,block=RepConv, basic_block=RepConv):
        super().__init__()
        self.conv1 = block(in_channels, out_channels)
        self.block = nn.Sequential(*(block(out_channels, out_channels) for _ in range(n - 1))) if n>1 else None
        if block == BottleRep:
            self.conv1 = BottleRep(in_channels, out_channels, basic_block=basic_block, weight=True)
            n = n // 2
            self.block = nn.Sequential(*(BottleRep(out_channels,out_channels,basic_block=basic_block, weight=True) for _ in range(n - 1)))

    def forward(self, x):
        x= self.conv1(x)
        if self.block is not None:
            x= self.block(x)
        return x


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class Silence(nn.Module):
    """Silence."""

    def __init__(self):
        """Initializes the Silence module."""
        super(Silence, self).__init__()

    def forward(self, x):
        """Forward pass through Silence layer."""
        return x


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super(CBLinear, self).__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        outs = self.conv(x).split(self.c2s, dim=1)
        return outs


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super(CBFuse, self).__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        out = torch.sum(torch.stack(res + xs[-1:]), dim=0)
        return out

class TopkRouting(nn.Module):
    """
    differentiable topk routing with scaling
    Args:
        qk_dim: int, feature dimension of query and key
        topk: int, the 'topk'
        qk_scale: int or None, temperature (multiply) of softmax activation
        with_param: bool, wether inorporate learnable params in routing unit
        diff_routing: bool, wether make routing differentiable
        soft_routing: bool, wether make output value multiplied by routing weights
    """

    def __init__(self, qk_dim, topk=4, qk_scale=None, param_routing=False, diff_routing=False):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim ** -0.5
        self.diff_routing = diff_routing
        # TODO: norm layer before/after linear?
        self.emb = nn.Linear(qk_dim, qk_dim) if param_routing else nn.Identity()
        # routing activation
        self.routing_act = nn.Softmax(dim=-1)

    def forward(self, query: Tensor, key: Tensor) -> Tuple[Tensor]:
        """
        Args:
            q, k: (n, p^2, c) tensor
        Return:
            r_weight, topk_index: (n, p^2, topk) tensor
        """
        if not self.diff_routing:
            query, key = query.detach(), key.detach()
        query_hat, key_hat = self.emb(query), self.emb(key)  # per-window pooling -> (n, p^2, c)
        attn_logit = (query_hat * self.scale) @ key_hat.transpose(-2, -1)  # (n, p^2, p^2)
        topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)  # (n, p^2, k), (n, p^2, k)
        r_weight = self.routing_act(topk_attn_logit)  # (n, p^2, k)

        return r_weight, topk_index


class KVGather(nn.Module):
    def __init__(self, mul_weight='none'):
        super().__init__()
        assert mul_weight in ['none', 'soft', 'hard']
        self.mul_weight = mul_weight

    def forward(self, r_idx: Tensor, r_weight: Tensor, kv: Tensor):
        """
        r_idx: (n, p^2, topk) tensor
        r_weight: (n, p^2, topk) tensor
        kv: (n, p^2, w^2, c_kq+c_v)
        Return:
            (n, p^2, topk, w^2, c_kq+c_v) tensor
        """
        # select kv according to routing index
        n, p2, w2, c_kv = kv.size()
        topk = r_idx.size(-1)
        # print(r_idx.size(), r_weight.size())
        # FIXME: gather consumes much memory (topk times redundancy), write cuda kernel?
        topk_kv = torch.gather(kv.view(n, 1, p2, w2, c_kv).expand(-1, p2, -1, -1, -1),
                               # (n, p^2, p^2, w^2, c_kv) without mem cpy
                               dim=2,
                               index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_kv)
                               # (n, p^2, k, w^2, c_kv)
                               )

        if self.mul_weight == 'soft':
            topk_kv = r_weight.view(n, p2, topk, 1, 1) * topk_kv  # (n, p^2, k, w^2, c_kv)
        elif self.mul_weight == 'hard':
            raise NotImplementedError('differentiable hard routing TBA')
        # else: #'none'
        #     topk_kv = topk_kv # do nothing

        return topk_kv


class QKVLinear(nn.Module):
    def __init__(self, dim, qk_dim, bias=True):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.qkv = nn.Linear(dim, qk_dim + qk_dim + dim, bias=bias)

    def forward(self, x):
        q, kv = self.qkv(x).split([self.qk_dim, self.qk_dim + self.dim], dim=-1)
        return q, kv
        # q, k, v = self.qkv(x).split([self.qk_dim, self.qk_dim, self.dim], dim=-1)
        # return q, k, v

class BiLevelRoutingAttention(nn.Module):
    """
    n_win: number of windows in one side (so the actual number of windows is n_win*n_win)
    kv_per_win: for kv_downsample_mode='ada_xxxpool' only, number of key/values per window. Similar to n_win, the actual number is kv_per_win*kv_per_win.
    topk: topk for window filtering
    param_attention: 'qkvo'-linear for q,k,v and o, 'none': param free attention
    param_routing: extra linear for routing
    diff_routing: wether to set routing differentiable
    soft_routing: wether to multiply soft routing weights
    """

    ###参数可自行修改
    def __init__(self, dim, n_win=7, num_heads=8, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='identity',
                 topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False,
                 side_dwconv=3,
                 auto_pad=True):
        super().__init__()
        # local attention setting
        self.dim = dim
        self.n_win = n_win  # Wh, Ww
        self.num_heads = num_heads
        self.qk_dim = qk_dim or dim
        assert self.qk_dim % num_heads == 0 and self.dim % num_heads == 0, 'qk_dim and dim must be divisible by num_heads!'
        self.scale = qk_scale or self.qk_dim ** -0.5

        ################side_dwconv (i.e. LCE in ShuntedTransformer)###########
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv // 2,
                              groups=dim) if side_dwconv > 0 else \
            lambda x: torch.zeros_like(x)

        ################ global routing setting #################
        self.topk = topk
        self.param_routing = param_routing
        self.diff_routing = diff_routing
        self.soft_routing = soft_routing
        # router
        assert not (self.param_routing and not self.diff_routing)  # cannot be with_param=True and diff_routing=False
        self.router = TopkRouting(qk_dim=self.qk_dim,
                                  qk_scale=self.scale,
                                  topk=self.topk,
                                  diff_routing=self.diff_routing,
                                  param_routing=self.param_routing)
        if self.soft_routing:  # soft routing, always diffrentiable (if no detach)
            mul_weight = 'soft'
        elif self.diff_routing:  # hard differentiable routing
            mul_weight = 'hard'
        else:  # hard non-differentiable routing
            mul_weight = 'none'
        self.kv_gather = KVGather(mul_weight=mul_weight)

        # qkv mapping (shared by both global routing and local attention)
        self.param_attention = param_attention
        if self.param_attention == 'qkvo':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Linear(dim, dim)
        elif self.param_attention == 'qkv':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Identity()
        else:
            raise ValueError(f'param_attention mode {self.param_attention} is not surpported!')

        self.kv_downsample_mode = kv_downsample_mode
        self.kv_per_win = kv_per_win
        self.kv_downsample_ratio = kv_downsample_ratio
        self.kv_downsample_kenel = kv_downsample_kernel
        if self.kv_downsample_mode == 'ada_avgpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveAvgPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'ada_maxpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveMaxPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'maxpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.MaxPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'avgpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.AvgPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'identity':  # no kv downsampling
            self.kv_down = nn.Identity()
        elif self.kv_downsample_mode == 'fracpool':
            # assert self.kv_downsample_ratio is not None
            # assert self.kv_downsample_kenel is not None
            # TODO: fracpool
            # 1. kernel size should be input size dependent
            # 2. there is a random factor, need to avoid independent sampling for k and v
            raise NotImplementedError('fracpool policy is not implemented yet!')
        elif kv_downsample_mode == 'conv':
            # TODO: need to consider the case where k != v so that need two downsample modules
            raise NotImplementedError('conv policy is not implemented yet!')
        else:
            raise ValueError(f'kv_down_sample_mode {self.kv_downsaple_mode} is not surpported!')

        # softmax for local attention
        self.attn_act = nn.Softmax(dim=-1)

        self.auto_pad = auto_pad

    def forward(self, x, ret_attn_mask=False):
        """
        x: NHWC tensor
        Return:
            NHWC tensor
        """
        x = rearrange(x, "n c h w -> n h w c")
        # NOTE: use padding for semantic segmentation
        ###################################################
        if self.auto_pad:
            N, H_in, W_in, C = x.size()

            pad_l = pad_t = 0
            pad_r = (self.n_win - W_in % self.n_win) % self.n_win
            pad_b = (self.n_win - H_in % self.n_win) % self.n_win
            x = F.pad(x, (0, 0,  # dim=-1
                          pad_l, pad_r,  # dim=-2
                          pad_t, pad_b))  # dim=-3
            _, H, W, _ = x.size()  # padded size
        else:
            N, H, W, C = x.size()
            assert H % self.n_win == 0 and W % self.n_win == 0  #
        ###################################################

        # patchify, (n, p^2, w, w, c), keep 2d window as we need 2d pooling to reduce kv size
        x = rearrange(x, "n (j h) (i w) c -> n (j i) h w c", j=self.n_win, i=self.n_win)

        #################qkv projection###################
        # q: (n, p^2, w, w, c_qk)
        # kv: (n, p^2, w, w, c_qk+c_v)
        # NOTE: separte kv if there were memory leak issue caused by gather
        q, kv = self.qkv(x)

        # pixel-wise qkv
        # q_pix: (n, p^2, w^2, c_qk)
        # kv_pix: (n, p^2, h_kv*w_kv, c_qk+c_v)
        q_pix = rearrange(q, 'n p2 h w c -> n p2 (h w) c')
        kv_pix = self.kv_down(rearrange(kv, 'n p2 h w c -> (n p2) c h w'))
        kv_pix = rearrange(kv_pix, '(n j i) c h w -> n (j i) (h w) c', j=self.n_win, i=self.n_win)

        q_win, k_win = q.mean([2, 3]), kv[..., 0:self.qk_dim].mean(
            [2, 3])  # window-wise qk, (n, p^2, c_qk), (n, p^2, c_qk)

        ##################side_dwconv(lepe)##################
        # NOTE: call contiguous to avoid gradient warning when using ddp
        lepe = self.lepe(rearrange(kv[..., self.qk_dim:], 'n (j i) h w c -> n c (j h) (i w)', j=self.n_win,
                                   i=self.n_win).contiguous())
        lepe = rearrange(lepe, 'n c (j h) (i w) -> n (j h) (i w) c', j=self.n_win, i=self.n_win)

        ############ gather q dependent k/v #################

        r_weight, r_idx = self.router(q_win, k_win)  # both are (n, p^2, topk) tensors

        kv_pix_sel = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kv_pix)  # (n, p^2, topk, h_kv*w_kv, c_qk+c_v)
        k_pix_sel, v_pix_sel = kv_pix_sel.split([self.qk_dim, self.dim], dim=-1)
        # kv_pix_sel: (n, p^2, topk, h_kv*w_kv, c_qk)
        # v_pix_sel: (n, p^2, topk, h_kv*w_kv, c_v)

        ######### do attention as normal ####################
        k_pix_sel = rearrange(k_pix_sel, 'n p2 k w2 (m c) -> (n p2) m c (k w2)',
                              m=self.num_heads)  # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_kq//m) transpose here?
        v_pix_sel = rearrange(v_pix_sel, 'n p2 k w2 (m c) -> (n p2) m (k w2) c',
                              m=self.num_heads)  # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_v//m)
        q_pix = rearrange(q_pix, 'n p2 w2 (m c) -> (n p2) m w2 c',
                          m=self.num_heads)  # to BMLC tensor (n*p^2, m, w^2, c_qk//m)

        # param-free multihead attention
        attn_weight = (
                                  q_pix * self.scale) @ k_pix_sel  # (n*p^2, m, w^2, c) @ (n*p^2, m, c, topk*h_kv*w_kv) -> (n*p^2, m, w^2, topk*h_kv*w_kv)
        attn_weight = self.attn_act(attn_weight)
        out = attn_weight @ v_pix_sel  # (n*p^2, m, w^2, topk*h_kv*w_kv) @ (n*p^2, m, topk*h_kv*w_kv, c) -> (n*p^2, m, w^2, c)
        out = rearrange(out, '(n j i) m (h w) c -> n (j h) (i w) (m c)', j=self.n_win, i=self.n_win,
                        h=H // self.n_win, w=W // self.n_win)

        out = out + lepe
        # output linear
        out = self.wo(out)

        # NOTE: use padding for semantic segmentation
        # crop padded region
        if self.auto_pad and (pad_r > 0 or pad_b > 0):
            out = out[:, :H_in, :W_in, :].contiguous()

        if ret_attn_mask:
            return out, r_weight, r_idx, attn_weight
        else:
            return rearrange(out, "n h w c -> n c h w")

class BiLevelRoutingAttention(nn.Module):
    """
    n_win: number of windows in one side (so the actual number of windows is n_win*n_win)
    kv_per_win: for kv_downsample_mode='ada_xxxpool' only, number of key/values per window. Similar to n_win, the actual number is kv_per_win*kv_per_win.
    topk: topk for window filtering
    param_attention: 'qkvo'-linear for q,k,v and o, 'none': param free attention
    param_routing: extra linear for routing
    diff_routing: wether to set routing differentiable
    soft_routing: wether to multiply soft routing weights
    """

    ###参数可自行修改
    def __init__(self, dim, n_win=7, num_heads=8, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='identity',
                 topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False,
                 side_dwconv=3,
                 auto_pad=True):
        super().__init__()
        # local attention setting
        self.dim = dim
        self.n_win = n_win  # Wh, Ww
        self.num_heads = num_heads
        self.qk_dim = qk_dim or dim
        assert self.qk_dim % num_heads == 0 and self.dim % num_heads == 0, 'qk_dim and dim must be divisible by num_heads!'
        self.scale = qk_scale or self.qk_dim ** -0.5

        ################side_dwconv (i.e. LCE in ShuntedTransformer)###########
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv // 2,
                              groups=dim) if side_dwconv > 0 else \
            lambda x: torch.zeros_like(x)

        ################ global routing setting #################
        self.topk = topk
        self.param_routing = param_routing
        self.diff_routing = diff_routing
        self.soft_routing = soft_routing
        # router
        assert not (self.param_routing and not self.diff_routing)  # cannot be with_param=True and diff_routing=False
        self.router = TopkRouting(qk_dim=self.qk_dim,
                                  qk_scale=self.scale,
                                  topk=self.topk,
                                  diff_routing=self.diff_routing,
                                  param_routing=self.param_routing)
        if self.soft_routing:  # soft routing, always diffrentiable (if no detach)
            mul_weight = 'soft'
        elif self.diff_routing:  # hard differentiable routing
            mul_weight = 'hard'
        else:  # hard non-differentiable routing
            mul_weight = 'none'
        self.kv_gather = KVGather(mul_weight=mul_weight)

        # qkv mapping (shared by both global routing and local attention)
        self.param_attention = param_attention
        if self.param_attention == 'qkvo':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Linear(dim, dim)
        elif self.param_attention == 'qkv':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Identity()
        else:
            raise ValueError(f'param_attention mode {self.param_attention} is not surpported!')

        self.kv_downsample_mode = kv_downsample_mode
        self.kv_per_win = kv_per_win
        self.kv_downsample_ratio = kv_downsample_ratio
        self.kv_downsample_kenel = kv_downsample_kernel
        if self.kv_downsample_mode == 'ada_avgpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveAvgPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'ada_maxpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveMaxPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'maxpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.MaxPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'avgpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.AvgPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'identity':  # no kv downsampling
            self.kv_down = nn.Identity()
        elif self.kv_downsample_mode == 'fracpool':
            # assert self.kv_downsample_ratio is not None
            # assert self.kv_downsample_kenel is not None
            # TODO: fracpool
            # 1. kernel size should be input size dependent
            # 2. there is a random factor, need to avoid independent sampling for k and v
            raise NotImplementedError('fracpool policy is not implemented yet!')
        elif kv_downsample_mode == 'conv':
            # TODO: need to consider the case where k != v so that need two downsample modules
            raise NotImplementedError('conv policy is not implemented yet!')
        else:
            raise ValueError(f'kv_down_sample_mode {self.kv_downsaple_mode} is not surpported!')

        # softmax for local attention
        self.attn_act = nn.Softmax(dim=-1)

        self.auto_pad = auto_pad

    def forward(self, x, ret_attn_mask=False):
        """
        x: NHWC tensor
        Return:
            NHWC tensor
        """
        x = rearrange(x, "n c h w -> n h w c")
        # NOTE: use padding for semantic segmentation
        ###################################################
        if self.auto_pad:
            N, H_in, W_in, C = x.size()

            pad_l = pad_t = 0
            pad_r = (self.n_win - W_in % self.n_win) % self.n_win
            pad_b = (self.n_win - H_in % self.n_win) % self.n_win
            x = F.pad(x, (0, 0,  # dim=-1
                          pad_l, pad_r,  # dim=-2
                          pad_t, pad_b))  # dim=-3
            _, H, W, _ = x.size()  # padded size
        else:
            N, H, W, C = x.size()
            assert H % self.n_win == 0 and W % self.n_win == 0  #
        ###################################################

        # patchify, (n, p^2, w, w, c), keep 2d window as we need 2d pooling to reduce kv size
        x = rearrange(x, "n (j h) (i w) c -> n (j i) h w c", j=self.n_win, i=self.n_win)

        #################qkv projection###################
        # q: (n, p^2, w, w, c_qk)
        # kv: (n, p^2, w, w, c_qk+c_v)
        # NOTE: separte kv if there were memory leak issue caused by gather
        q, kv = self.qkv(x)

        # pixel-wise qkv
        # q_pix: (n, p^2, w^2, c_qk)
        # kv_pix: (n, p^2, h_kv*w_kv, c_qk+c_v)
        q_pix = rearrange(q, 'n p2 h w c -> n p2 (h w) c')
        kv_pix = self.kv_down(rearrange(kv, 'n p2 h w c -> (n p2) c h w'))
        kv_pix = rearrange(kv_pix, '(n j i) c h w -> n (j i) (h w) c', j=self.n_win, i=self.n_win)

        q_win, k_win = q.mean([2, 3]), kv[..., 0:self.qk_dim].mean(
            [2, 3])  # window-wise qk, (n, p^2, c_qk), (n, p^2, c_qk)

        ##################side_dwconv(lepe)##################
        # NOTE: call contiguous to avoid gradient warning when using ddp
        lepe = self.lepe(rearrange(kv[..., self.qk_dim:], 'n (j i) h w c -> n c (j h) (i w)', j=self.n_win,
                                   i=self.n_win).contiguous())
        lepe = rearrange(lepe, 'n c (j h) (i w) -> n (j h) (i w) c', j=self.n_win, i=self.n_win)

        ############ gather q dependent k/v #################

        r_weight, r_idx = self.router(q_win, k_win)  # both are (n, p^2, topk) tensors

        kv_pix_sel = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kv_pix)  # (n, p^2, topk, h_kv*w_kv, c_qk+c_v)
        k_pix_sel, v_pix_sel = kv_pix_sel.split([self.qk_dim, self.dim], dim=-1)
        # kv_pix_sel: (n, p^2, topk, h_kv*w_kv, c_qk)
        # v_pix_sel: (n, p^2, topk, h_kv*w_kv, c_v)

        ######### do attention as normal ####################
        k_pix_sel = rearrange(k_pix_sel, 'n p2 k w2 (m c) -> (n p2) m c (k w2)',
                              m=self.num_heads)  # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_kq//m) transpose here?
        v_pix_sel = rearrange(v_pix_sel, 'n p2 k w2 (m c) -> (n p2) m (k w2) c',
                              m=self.num_heads)  # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_v//m)
        q_pix = rearrange(q_pix, 'n p2 w2 (m c) -> (n p2) m w2 c',
                          m=self.num_heads)  # to BMLC tensor (n*p^2, m, w^2, c_qk//m)

        # param-free multihead attention
        attn_weight = (
                                  q_pix * self.scale) @ k_pix_sel  # (n*p^2, m, w^2, c) @ (n*p^2, m, c, topk*h_kv*w_kv) -> (n*p^2, m, w^2, topk*h_kv*w_kv)
        attn_weight = self.attn_act(attn_weight)
        out = attn_weight @ v_pix_sel  # (n*p^2, m, w^2, topk*h_kv*w_kv) @ (n*p^2, m, topk*h_kv*w_kv, c) -> (n*p^2, m, w^2, c)
        out = rearrange(out, '(n j i) m (h w) c -> n (j h) (i w) (m c)', j=self.n_win, i=self.n_win,
                        h=H // self.n_win, w=W // self.n_win)

        out = out + lepe
        # output linear
        out = self.wo(out)

        # NOTE: use padding for semantic segmentation
        # crop padded region
        if self.auto_pad and (pad_r > 0 or pad_b > 0):
            out = out[:, :H_in, :W_in, :].contiguous()

        if ret_attn_mask:
            return out, r_weight, r_idx, attn_weight
        else:
            return rearrange(out, "n h w c -> n c h w")
class Add(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sum(torch.stack(x, dim=0), dim=0)

class Zoom_cat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        l, m, s = x[0], x[1], x[2]
        tgt_size = m.shape[2:]
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        s = F.interpolate(s, m.shape[2:], mode='nearest')
        lms = torch.cat([l, m, s], dim=1)
        return lms

class ScalSeq(nn.Module):
    def __init__(self, inc, channel):
        super(ScalSeq, self).__init__()
        if channel != inc[0]:
            self.conv0 = Conv(inc[0], channel,1)
        self.conv1 =  Conv(inc[1], channel,1)
        self.conv2 =  Conv(inc[2], channel,1)
        self.conv3d = nn.Conv3d(channel,channel,kernel_size=(1,1,1))
        self.bn = nn.BatchNorm3d(channel)
        self.act = nn.LeakyReLU(0.1)
        self.pool_3d = nn.MaxPool3d(kernel_size=(3,1,1))

    def forward(self, x):
        p3, p4, p5 = x[0],x[1],x[2]
        if hasattr(self, 'conv0'):
            p3 = self.conv0(p3)
        p4_2 = self.conv1(p4)
        p4_2 = F.interpolate(p4_2, p3.size()[2:], mode='nearest')
        p5_2 = self.conv2(p5)
        p5_2 = F.interpolate(p5_2, p3.size()[2:], mode='nearest')
        p3_3d = torch.unsqueeze(p3, -3)
        p4_3d = torch.unsqueeze(p4_2, -3)
        p5_3d = torch.unsqueeze(p5_2, -3)
        combine = torch.cat([p3_3d, p4_3d, p5_3d],dim = 2)
        conv_3d = self.conv3d(combine)
        bn = self.bn(conv_3d)
        act = self.act(bn)
        x = self.pool_3d(act)
        x = torch.squeeze(x, 2)
        return x


class Faster_Block_EMA(nn.Module):
    def __init__(self,
                 inc,
                 dim,
                 n_div=4,
                 mlp_ratio=2,
                 drop_path=0.1,
                 layer_scale_init_value=0.0,
                 pconv_fw_type='split_cat'
                 ):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer = [
            Conv(dim, mlp_hidden_dim, 1),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )
        self.attention = EMA(dim)

        self.adjust_channel = None
        if inc != dim:
            self.adjust_channel = Conv(inc, dim, 1)

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x):
        if self.adjust_channel is not None:
            x = self.adjust_channel(x)
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.attention(self.drop_path(self.mlp(x)))
        return x

    def forward_layer_scale(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x
class C2f_Faster_EMA(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Faster_Block_EMA(self.c, self.c) for _ in range(n))



class Bottleneck_DBB(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DiverseBranchBlock(c1, c_, k[0], 1)
        self.cv2 = DiverseBranchBlock(c_, c2, k[1], 1, groups=g)
class C2f_DBB(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_DBB(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

# gfpn
class CSPStage(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 n,
                 block_fn='BasicBlock_3x3_Reverse',
                 ch_hidden_ratio=1.0,
                 act='silu',
                 spp=False):
        super(CSPStage, self).__init__()

        split_ratio = 2
        ch_first = int(ch_out // split_ratio)
        ch_mid = int(ch_out - ch_first)
        self.conv1 = Conv(ch_in, ch_first, 1)
        self.conv2 = Conv(ch_in, ch_mid, 1)
        self.convs = nn.Sequential()

        next_ch_in = ch_mid
        for i in range(n):
            if block_fn == 'BasicBlock_3x3_Reverse':
                self.convs.add_module(
                    str(i),
                    BasicBlock_3x3_Reverse(next_ch_in,
                                           ch_hidden_ratio,
                                           ch_mid,
                                           shortcut=True))
            else:
                raise NotImplementedError
            if i == (n - 1) // 2 and spp:
                self.convs.add_module('spp', SPP(ch_mid * 4, ch_mid, 1, [5, 9, 13]))
            next_ch_in = ch_mid
        self.conv3 = Conv(ch_mid * n + ch_first, ch_out, 1)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)

        mid_out = [y1]
        for conv in self.convs:
            y2 = conv(y2)
            mid_out.append(y2)
        y = torch.cat(mid_out, axis=1)
        y = self.conv3(y)
        return y

class BasicBlock_3x3_Reverse(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_hidden_ratio,
                 ch_out,
                 shortcut=True):
        super(BasicBlock_3x3_Reverse, self).__init__()
        assert ch_in == ch_out
        ch_hidden = int(ch_in * ch_hidden_ratio)
        self.conv1 = Conv(ch_hidden, ch_out, 3, s=1)
        self.conv2 = RepConv(ch_in, ch_hidden, 3, s=1)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv2(x)
        y = self.conv1(y)
        if self.shortcut:
            return x + y
        else:
            return y



class GSConv(nn.Module):
    # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, p, g, d, Conv.default_act)
        self.cv2 = Conv(c_, c_, 5, 1, p, c_, d, Conv.default_act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # shuffle
        # y = x2.reshape(x2.shape[0], 2, x2.shape[1] // 2, x2.shape[2], x2.shape[3])
        # y = y.permute(0, 2, 1, 3, 4)
        # return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])

        b, n, h, w = x2.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)

        return torch.cat((y[0], y[1]), 1)


class GSConvns(GSConv):
    # GSConv with a normative-shuffle https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__(c1, c2, k, s, p, g, act=True)
        c_ = c2 // 2
        self.shuf = nn.Conv2d(c_ * 2, c2, 1, 1, 0, bias=False)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # normative-shuffle, TRT supported
        return nn.ReLU()(self.shuf(x2))


class GSBottleneck(nn.Module):
    # GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=3, s=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        # for lighting
        self.conv_lighting = nn.Sequential(
            GSConv(c1, c_, 1, 1),
            GSConv(c_, c2, 3, 1, act=False))
        self.shortcut = Conv(c1, c2, 1, 1, act=False)

    def forward(self, x):
        return self.conv_lighting(x) + self.shortcut(x)


class GSBottleneckns(GSBottleneck):
    # GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=3, s=1, e=0.5):
        super().__init__(c1, c2, k, s, e)
        c_ = int(c2 * e)
        # for lighting
        self.conv_lighting = nn.Sequential(
            GSConvns(c1, c_, 1, 1),
            GSConvns(c_, c2, 3, 1, act=False))


class GSBottleneckC(GSBottleneck):
    # cheap GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__(c1, c2, k, s)
        self.shortcut = DWConv(c1, c2, k, s, act=False)


class VoVGSCSP(nn.Module):
    # VoVGSCSP module with GSBottleneck
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.gsb = nn.Sequential(*(GSBottleneck(c_, c_, e=1.0) for _ in range(n)))
        self.res = Conv(c_, c_, 3, 1, act=False)
        self.cv3 = Conv(2 * c_, c2, 1)

    def forward(self, x):
        x1 = self.gsb(self.cv1(x))
        y = self.cv2(x)
        return self.cv3(torch.cat((y, x1), dim=1))


class VoVGSCSPns(VoVGSCSP):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.gsb = nn.Sequential(*(GSBottleneckns(c_, c_, e=1.0) for _ in range(n)))


class VoVGSCSPC(VoVGSCSP):
    # cheap VoVGSCSP module with GSBottleneck
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2)
        c_ = int(c2 * 0.5)  # hidden channels
        self.gsb = GSBottleneckC(c_, c_, 1, 1)

import torch.nn.functional as F
class SDI(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # self.convs = nn.ModuleList([nn.Conv2d(channel, channels[0], kernel_size=3, stride=1, padding=1) for channel in channels])
        self.convs = nn.ModuleList([GSConv(channel, channels[0]) for channel in channels])

    def forward(self, xs):
        ans = torch.ones_like(xs[0])
        target_size = xs[0].shape[2:]
        for i, x in enumerate(xs):
            if x.shape[-1] > target_size[-1]:
                x = F.adaptive_avg_pool2d(x, (target_size[0], target_size[1]))
            elif x.shape[-1] < target_size[-1]:
                x = F.interpolate(x, size=(target_size[0], target_size[1]),
                                      mode='bilinear', align_corners=True)
            ans = ans * self.convs[i](x)
        return ans


class Fusion(nn.Module):
    def __init__(self, inc_list, fusion='bifpn') -> None:
        super().__init__()

        assert fusion in ['weight', 'adaptive', 'concat', 'bifpn', 'SDI']
        self.fusion = fusion

        if self.fusion == 'bifpn':
            self.fusion_weight = nn.Parameter(torch.ones(len(inc_list), dtype=torch.float32), requires_grad=True)
            self.relu = nn.ReLU()
            self.epsilon = 1e-4
        elif self.fusion == 'SDI':
            self.SDI = SDI(inc_list)
        else:
            self.fusion_conv = nn.ModuleList([Conv(inc, inc, 1) for inc in inc_list])

            if self.fusion == 'adaptive':
                self.fusion_adaptive = Conv(sum(inc_list), len(inc_list), 1)

    def forward(self, x):
        if self.fusion in ['weight', 'adaptive']:
            for i in range(len(x)):
                x[i] = self.fusion_conv[i](x[i])
        if self.fusion == 'weight':
            return torch.sum(torch.stack(x, dim=0), dim=0)
        elif self.fusion == 'adaptive':
            fusion = torch.softmax(self.fusion_adaptive(torch.cat(x, dim=1)), dim=1)
            x_weight = torch.split(fusion, [1] * len(x), dim=1)
            return torch.sum(torch.stack([x_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)
        elif self.fusion == 'concat':
            return torch.cat(x, dim=1)
        elif self.fusion == 'bifpn':
            fusion_weight = self.relu(self.fusion_weight.clone())
            fusion_weight = fusion_weight / (torch.sum(fusion_weight, dim=0))
            return torch.sum(torch.stack([fusion_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)
        elif self.fusion == 'SDI':
            return self.SDI(x)


####################################### Focus Diffusion Pyramid Network end ########################################

class FocusFeature(nn.Module):
    def __init__(self, inc, kernel_sizes=(5, 7, 9, 11), e=0.5) -> None:
        super().__init__()
        hidc = int(inc[1] * e)

        self.conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv(inc[0], hidc, 1)
        )
        self.conv2 = Conv(inc[1], hidc, 1) if e != 1 else nn.Identity()
        self.conv3 = ADown(inc[2], hidc)

        self.dw_conv = nn.ModuleList(
            nn.Conv2d(hidc * 3, hidc * 3, kernel_size=k, padding=autopad(k), groups=hidc * 3) for k in kernel_sizes)
        self.pw_conv = Conv(hidc * 3, hidc * 3)

    def forward(self, x):
        x1, x2, x3 = x
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)

        x = torch.cat([x1, x2, x3], dim=1)
        feature = torch.sum(torch.stack([x] + [layer(x) for layer in self.dw_conv], dim=0), dim=0)
        feature = self.pw_conv(feature)

        x = x + feature
        return x


######################################## PKINet start ########################################

class GSiLU(nn.Module):
    """Global Sigmoid-Gated Linear Unit, reproduced from paper <SIMPLE CNN FOR VISION>"""

    def __init__(self):
        super().__init__()
        self.adpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return x * torch.sigmoid(self.adpool(x))


class PKIModule_CAA(nn.Module):
    def __init__(self, ch, h_kernel_size=11, v_kernel_size=11) -> None:
        super().__init__()

        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = Conv(ch, ch)
        self.h_conv = nn.Conv2d(ch, ch, (1, h_kernel_size), 1, (0, h_kernel_size // 2), 1, ch)
        self.v_conv = nn.Conv2d(ch, ch, (v_kernel_size, 1), 1, (v_kernel_size // 2, 0), 1, ch)
        self.conv2 = Conv(ch, ch)
        self.act = nn.Sigmoid()

    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return attn_factor


class PKIModule(nn.Module):
    def __init__(self, inc, ouc, kernel_sizes=(3, 5, 7, 9, 11), expansion=1.0, with_caa=True, caa_kernel_size=11,
                 add_identity=True) -> None:
        super().__init__()
        hidc = make_divisible(int(ouc * expansion), 8)

        self.pre_conv = Conv(inc, hidc)
        self.dw_conv = nn.ModuleList(
            nn.Conv2d(hidc, hidc, kernel_size=k, padding=autopad(k), groups=hidc) for k in kernel_sizes)
        self.pw_conv = Conv(hidc, hidc)
        self.post_conv = Conv(hidc, ouc)

        if with_caa:
            self.caa_factor = PKIModule_CAA(hidc, caa_kernel_size, caa_kernel_size)
        else:
            self.caa_factor = None

        self.add_identity = add_identity and inc == ouc

    def forward(self, x):
        x = self.pre_conv(x)

        y = x
        x = self.dw_conv[0](x)
        x = torch.sum(torch.stack([x] + [layer(x) for layer in self.dw_conv[1:]], dim=0), dim=0)
        x = self.pw_conv(x)

        if self.caa_factor is not None:
            y = self.caa_factor(y)
        if self.add_identity:
            y = x * y
            x = x + y
        else:
            x = x * y

        x = self.post_conv(x)
        return x


class C3_PKIModule(C3):
    def __init__(self, c1, c2, n=1, kernel_sizes=(3, 5, 7, 9, 11), expansion=1.0, with_caa=True, caa_kernel_size=11,
                 add_identity=True, g=1, e=0.5):
        super().__init__(c1, c2, n, True, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(
            *(PKIModule(c_, c_, kernel_sizes, expansion, with_caa, caa_kernel_size, add_identity) for _ in range(n)))


class C2f_PKIModule(C2f):
    def __init__(self, c1, c2, n=1, kernel_sizes=(3, 5, 7, 9, 11), expansion=1.0, with_caa=True, caa_kernel_size=11,
                 add_identity=True, g=1, e=0.5):
        super().__init__(c1, c2, n, True, g, e)
        self.m = nn.ModuleList(
            PKIModule(self.c, self.c, kernel_sizes, expansion, with_caa, caa_kernel_size, add_identity) for _ in
            range(n))
class RepConvN(nn.Module):
    """RepConv is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process"""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process"""
        if hasattr(self, 'conv'):
            return self.forward_fuse(x)
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _avg_to_3x3_tensor(self, avgp):
        channels = self.c1
        groups = self.g
        kernel_size = avgp.kernel_size
        input_dim = channels // groups
        k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
        k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
        return k

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(in_channels=self.conv1.conv.in_channels,
                              out_channels=self.conv1.conv.out_channels,
                              kernel_size=self.conv1.conv.kernel_size,
                              stride=self.conv1.conv.stride,
                              padding=self.conv1.conv.padding,
                              dilation=self.conv1.conv.dilation,
                              groups=self.conv1.conv.groups,
                              bias=True).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
class RepNBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConvN(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
class RepNCSP(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(RepNBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class RepNCSPELAN4_CAA(nn.Module):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, c5=1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepNCSP(c3 // 2, c4, c5), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepNCSP(c4, c4, c5), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)
        self.caa = CAA(c3 + (2 * c4))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(self.caa(torch.cat(y, 1)))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(self.caa(torch.cat(y, 1)))

######################################## PKINet end ########################################


try:
    from mmcv.cnn import build_activation_layer, build_norm_layer
    from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d
    from mmengine.model import constant_init, normal_init
except ImportError as e:
    pass

class DyDCNv2(nn.Module):
    """ModulatedDeformConv2d with normalization layer used in DyHead.
    This module cannot be configured with `conv_cfg=dict(type='DCNv2')`
    because DyHead calculates offset and mask from middle-level feature.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int | tuple[int], optional): Stride of the convolution.
            Default: 1.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='GN', num_groups=16, requires_grad=True).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)):
        super().__init__()
        self.with_norm = norm_cfg is not None
        bias = not self.with_norm
        self.conv = ModulatedDeformConv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=bias)
        if self.with_norm:
            self.norm = build_norm_layer(norm_cfg, out_channels)[1]

    def forward(self, x, offset, mask):
        """Forward function."""
        x = self.conv(x.contiguous(), offset, mask)
        if self.with_norm:
            x = self.norm(x)
        return x

######################################## Context and Spatial Feature Calibration start ########################################

class SpatialAttention_CGA(nn.Module):
    def __init__(self):
        super(SpatialAttention_CGA, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention_CGA(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention_CGA, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn


class PixelAttention_CGA(nn.Module):
    def __init__(self, dim):
        super(PixelAttention_CGA, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = rearrange(x2, 'b c t h w -> b (c t) h w')
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention_CGA()
        self.ca = ChannelAttention_CGA(dim, reduction)
        self.pa = PixelAttention_CGA(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x, y = data
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result

######################################## Cross-Scale Mutil-Head Self-Attention start ########################################

class CSMHSA(nn.Module):
    def __init__(self, n_dims, heads=8):
        super(CSMHSA, self).__init__()

        self.heads = heads
        self.query = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(n_dims[0], n_dims[1], kernel_size=1)
        )
        self.key = nn.Conv2d(n_dims[1], n_dims[1], kernel_size=1)
        self.value = nn.Conv2d(n_dims[1], n_dims[1], kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_high, x_low = x
        n_batch, C, width, height = x_low.size()
        q = self.query(x_high).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x_low).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x_low).view(n_batch, self.heads, C // self.heads, -1)
        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)
        attention = self.softmax(content_content)
        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)
        return out


######################################## RepViT start ########################################

class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class RepViTBlock(nn.Module):
    def __init__(self, inp, oup, use_se=True):
        super(RepViTBlock, self).__init__()

        self.identity = inp == oup
        hidden_dim = 2 * inp

        self.token_mixer = nn.Sequential(
            RepVGGDW(inp),
            SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
        )
        self.channel_mixer = Residual(nn.Sequential(
            # pw
            Conv2d_BN(inp, hidden_dim, 1, 1, 0),
            nn.GELU(),
            # pw-linear
            Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
        ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))


class RepViTBlock_EMA(RepViTBlock):
    def __init__(self, inp, oup, use_se=True):
        super().__init__(inp, oup, use_se)

        self.token_mixer = nn.Sequential(
            RepVGGDW(inp),
            EMA(inp) if use_se else nn.Identity(),
        )


class C3_RVB(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepViTBlock(c_, c_, False) for _ in range(n)))


class C2f_RVB(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(RepViTBlock(self.c, self.c, False) for _ in range(n))


class C3_RVB_SE(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepViTBlock(c_, c_) for _ in range(n)))


class C2f_RVB_SE(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(RepViTBlock(self.c, self.c) for _ in range(n))


class C3_RVB_EMA(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepViTBlock_EMA(c_, c_) for _ in range(n)))


class C2f_RVB_EMA(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(RepViTBlock_EMA(self.c, self.c) for _ in range(n))

######################################## RepViT end ########################################

# CA
from mmcv.cnn import ConvModule
from mmengine.model import caffe2_xavier_init, constant_init
class ContextAggregation(nn.Module):
    """
    Context Aggregation Block.

    Args:
        in_channels (int): Number of input channels.
        reduction (int, optional): Channel reduction ratio. Default: 1.
        conv_cfg (dict or None, optional): Config dict for the convolution
            layer. Default: None.
    """

    def __init__(self, in_channels, reduction=1):
        super(ContextAggregation, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.inter_channels = max(in_channels // reduction, 1)

        conv_params = dict(kernel_size=1, act_cfg=None)

        self.a = ConvModule(in_channels, 1, **conv_params)
        self.k = ConvModule(in_channels, 1, **conv_params)
        self.v = ConvModule(in_channels, self.inter_channels, **conv_params)
        self.m = ConvModule(self.inter_channels, in_channels, **conv_params)

        self.init_weights()

    def init_weights(self):
        for m in (self.a, self.k, self.v):
            caffe2_xavier_init(m.conv)
        constant_init(self.m.conv, 0)

    def forward(self, x):
        n, c = x.size(0), self.inter_channels

        # a: [N, 1, H, W]
        a = self.a(x).sigmoid()

        # k: [N, 1, HW, 1]
        k = self.k(x).view(n, 1, -1, 1).softmax(2)

        # v: [N, 1, C, HW]
        v = self.v(x).view(n, 1, c, -1)

        # y: [N, C, 1, 1]
        y = torch.matmul(v, k).view(n, c, 1, 1)
        y = self.m(y) * a

        return x + y
