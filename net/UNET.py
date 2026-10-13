import torch.nn as nn 
import torch 
import numpy as np
import torch.nn.functional as F 
import enum
class DoubleConv(nn.Module):
    """UNet基础模块:两次卷积+BN+ReLU"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.double_conv(x)
class Down(nn.Module):
    """下采样模块：MaxPool + DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """上采样模块：转置卷积/双线性插值 + 特征融合"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels//2)
        else:
            # 转置卷积更高效但可能引入棋盘格伪影
            self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 特征图尺寸对齐（解决下采样时的尺寸不匹配）
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX//2, diffX - diffX//2, diffY//2, diffY - diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """输出层：1x1卷积映射到目标通道数"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNetDenoise(nn.Module):
    """完整去噪UNet网络"""
    def __init__(self, n_channels, n_classes=3, bilinear=False):
        super(UNetDenoise, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64,64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        # 输出与输入同维度，直接回归去噪结果
        return logits

    def enable_grad_debug(self):
        """ throws interuption error(匹配你的需求)"""
        for param in self.parameters():
            param.register_hook(lambda grad: torch.nan_to_num(grad, nan=0.0, posinf=1e5, neginf=-1e5))

            
'''
class StdType(enum.Enum):
    FIXED = 0
    CONSTANT = 1
    VARIABLE = 2


class   Gaussian_distribution(nn.Module):
     def __init__(self, in_size, out_size, std_type, init_std, init_output_scale=0.01):
        super().__init__()
        self._std_type = std_type

        self._build_params(in_size, out_size, init_std, init_output_scale)
        return
     def _build_param(self,in_size,out_size,init_std,init_output_scale=0.8):
        self._mean_net = torch.nn.Linear(in_size, out_size)
        torch.nn.init.uniform_(self._mean_net.weight, -init_output_scale, init_output_scale)
        torch.nn.init.zeros_(self._mean_net.bias)

        logstd = np.log(init_std)
        if (self._std_type == StdType.FIXED):
            self._logstd_net = torch.nn.Parameter(torch.zeros(out_size, requires_grad=False, dtype=torch.float32), requires_grad=False)
            torch.nn.init.constant_(self._logstd_net, logstd)
        elif (self._std_type == StdType.CONSTANT):
            self._logstd_net = torch.nn.Parameter(torch.zeros(out_size, requires_grad=True, dtype=torch.float32), requires_grad=True)
            torch.nn.init.constant_(self._logstd_net, logstd)
        elif (self._std_type == StdType.VARIABLE):
            self._logstd_net = torch.nn.Linear(in_size, out_size)
            torch.nn.init.uniform_(self._logstd_net.weight, -init_output_scale, init_output_scale) 
            torch.nn.init.constant_(self._logstd_net.bias, logstd)
        else:
            assert(False), "Unsupported StdType: {}".format(self._std_type)
     def forward(self, input):
        mean = self._mean_net(input)

        if (self._std_type == StdType.FIXED or self._std_type == StdType.CONSTANT):
            logstd = torch.broadcast_to(self._logstd_net, mean.shape)
        elif (self._std_type == StdType.VARIABLE):
            logstd = self._logstd_net(input)
        else:
            assert(False), "Unsupported StdType: {}".format(self._std_type)

        dist = Gaussian_distribution(mean=mean, logstd=logstd)
        return dist
'''