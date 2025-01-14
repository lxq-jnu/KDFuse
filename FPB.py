import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):

    def __init__(self, k_size=3):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        return y.expand_as(x)
class Enhance_module(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, r=2):
        super(Enhance_module, self).__init__()
        dim_inter  = dim_out
        dim_att = dim_inter // 4
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )


        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_inter, dim_att, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim_att),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_att, dim_att, 3, 1, padding=1, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_att),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_inter, dim_att, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim_att),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_att, dim_att, 3, 1, padding=2 * rate, dilation=2 * rate, bias=True),
            nn.BatchNorm2d(dim_att),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_inter, dim_att, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim_att),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_att, dim_att, 3, 1, padding=4 * rate, dilation=4 * rate, bias=True),
            nn.BatchNorm2d(dim_att),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_inter, dim_att, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim_att),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_att, dim_att, 3, 1, padding=8 * rate, dilation=8 * rate, bias=True),
            nn.BatchNorm2d(dim_att),
            nn.ReLU(inplace=True),
        )
        self.f2_conv = nn.Sequential(
            nn.Conv2d(dim_inter, dim_inter, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim_inter),
            nn.ReLU(inplace=True),
        )


        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_inter * 4, dim_out//2, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out//2),
            nn.ReLU(inplace=True),
        )

        self.ca=ChannelAttention()

    def forward(self, x):
        x = self.conv1(x)
        b, c, row, col = x.size()
        ch = c//2
        f = x
        conv3x3 = self.branch1(f)
        conv3x3_1 = self.branch2(f)
        conv3x3_2 = self.branch3(f)
        conv3x3_3 = self.branch4(f)

        feature_cat = torch.cat([conv3x3, conv3x3_1, conv3x3_2, conv3x3_3], dim=1)


        att_w = self.ca(feature_cat+f)
        result = f * att_w + f
        return result

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=6, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv1x1 = nn.Conv2d(dim, dim//2, kernel_size=1)


    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        attn = self.conv1x1(attn)
        return attn




class SpatialAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatialAttention, self).__init__()
        inter_channels = in_channels // 4
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(int(inter_channels)),
            nn.ReLU(inplace=True)
        )
        self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1,None)), nn.AdaptiveAvgPool2d((None,1))

        self.conv2 = nn.Sequential(
            nn.Conv2d(int(inter_channels), out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x_w, x_h = self.pool_w(x), self.pool_h(x)
        x = torch.matmul(x_h, x_w)
        
        out = self.conv2(x)

        return out
    


class SIM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SIM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.attn = AttentionModule(2*out_channels)
        self.ca = ChannelAttention()
        self.sigmoid = nn.Sigmoid()
        self.sa = SpatialAttention(out_channels, out_channels)

    def forward(self, seg, fu):
        seg = self.conv(seg)
        x = torch.cat((seg, fu),dim=1)
        w = self.sigmoid(self.attn(x))
        s1 = w * fu
        s1 = self.sa(s1) * s1
        s2 = (1-w) * seg
        s2 = fu * self.ca(s2)

        out = s1 + s2

        return out


