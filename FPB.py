import torch
import torch.nn as nn
import torch.nn.functional as F


# class ChannelAttention(nn.Module):

#     def __init__(self, in_channels, r=4):
#         super(ChannelAttention, self).__init__()
#         inter_channels = int(in_channels // r)
#         self.fc1 = nn.Conv2d(in_channels=in_channels, out_channels=inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
#         self.fc2 = nn.Conv2d(in_channels=inter_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=True)
#         self.in_channels = in_channels

#     def forward(self, inputs):
#         x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
#         # print('x:', x.shape)
#         x1 = self.fc1(x1)
#         x1 = F.relu(x1, inplace=True)
#         x1 = self.fc2(x1)
#         x1 = torch.sigmoid(x1)
#         x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
#         # print('x:', x.shape)
#         x2 = self.fc1(x2)
#         x2 = F.relu(x2, inplace=True)
#         x2 = self.fc2(x2)
#         x2 = torch.sigmoid(x2)
#         x = x1 + x2
#         x = x.view(-1, self.in_channels, 1, 1)
#         return x
class ChannelAttention(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, k_size=3):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return y.expand_as(x)
class Enhance_module(nn.Module):                       ##加入通道注意力机制
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
        # self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        # self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        # self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_inter * 4, dim_out//2, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out//2),
            nn.ReLU(inplace=True),
        )
        # print('dim_in:',dim_in)
        # print('dim_out:',dim_out)
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

        # result = f * att_w + feature_cat
        # print('result:',result.shape)
        return result
    # def forward(self, x):
    #     x = self.conv1(x)
    #     b, c, row, col = x.size()
    #     ch = c//2
    #     res = x
    #     f1 = x[:,:ch,:,:]
    #     f2 = x[:,ch:,:,:]
    #     conv3x3 = self.branch1(f1)
    #     conv3x3_1 = self.branch2(f1)
    #     conv3x3_2 = self.branch3(f1)
    #     conv3x3_3 = self.branch4(f1)
    #
    #     feature_cat = torch.cat([conv3x3, conv3x3_1, conv3x3_2,conv3x3_3, f2], dim=1)
    #     # print('feature:',feature_cat.shape)
    #     seaspp1=self.ca(feature_cat)              #加入通道注意力机制,权重
    #     # print('seaspp1:',seaspp1.shape)
    #
    #     se_feature_cat = seaspp1 * feature_cat
    #     result = se_feature_cat + res
    #     # print('result:',result.shape)
    #     return result
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


class AttentionModule(nn.Module):  #train cat
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)#深度卷积
        self.conv_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=6, groups=dim, dilation=3)#深度空洞卷积
        self.conv1 = nn.Conv2d(dim, dim, 1)#逐点卷积
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
    


class SIM_v2(nn.Module):
    def __init__(self, in_channels, out_channels): #inchannels 语义的通道 outchannels 融合的通道
        super(SIM_v2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

        )
        self.attn = AttentionModule(2*out_channels)# train cat
        self.ca = ChannelAttention()
        # self.sa = nn.Sequential(
        #     nn.Conv2d(out_channels, int(width), kernel_size=7, padding=3),
        #     nn.BatchNorm2d(int(width)),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(int(width), out_channels, kernel_size=7, padding=3),
        #     nn.BatchNorm2d(out_channels)
        # )


        self.sigmoid = nn.Sigmoid()
        self.sa = SpatialAttention(out_channels, out_channels)

    def forward(self, seg, fu):
        seg = self.conv(seg)
        x = torch.cat((seg, fu),dim=1)   #cat train
        w = self.sigmoid(self.attn(x))
        s1 = w * fu
        s1 = self.sa(s1) * s1

        s2 = (1-w) * seg
        s2 = fu * self.ca(s2)
        # s2 = s2 * self.ca(s2)

        
        out = s1 + s2

        return out

class SIM_DI(nn.Module):
    def __init__(self, in_channels, out_channels): #inchannels 语义的通道 outchannels 融合的通道
        super(SIM_DI, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

        )
        self.attn = AttentionModule(2*out_channels)# train cat
        self.ca = ChannelAttention()
        self.conv1x1 = nn.Conv2d(out_channels *2 , out_channels, kernel_size=1)
        # self.sa = nn.Sequential(
        #     nn.Conv2d(out_channels, int(width), kernel_size=7, padding=3),
        #     nn.BatchNorm2d(int(width)),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(int(width), out_channels, kernel_size=7, padding=3),
        #     nn.BatchNorm2d(out_channels)
        # )


        self.sigmoid = nn.Sigmoid()
        self.sa = SpatialAttention(out_channels, out_channels)

    def forward(self, seg, fu):
        seg = self.conv(seg)
        x = torch.cat((seg, fu),dim=1)   #cat train
        x = self.conv1x1(x)
        # w = self.sigmoid(self.attn(x))
        # s1 = w * fu
        # s1 = self.sa(s1) * s1

        # s2 = (1-w) * seg
        # s2 = s2 * self.ca(s2)

        
        out = x

        return out
    

