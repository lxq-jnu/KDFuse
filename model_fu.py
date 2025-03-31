import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch import nn
from FPB import Enhance_module,SIM_v2,SIM_DI
import argparse
import os
import torch.nn.functional as F




def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        act_func = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        act_func = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        act_func = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    # TODO: 新增silu和gelu激活函数
    elif act_type == 'silu':
        pass
    elif act_type == 'gelu':
        pass
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return act_func


class SeqConv3x3(nn.Module):
    def __init__(self, seq_type, inp_planes, out_planes):
        super(SeqConv3x3, self).__init__()

        self.type = seq_type
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        if self.type == 'conv1x1-sobelx':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(bias)
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 1, 0] = 2.0
                self.mask[i, 0, 2, 0] = 1.0
                self.mask[i, 0, 0, 2] = -1.0
                self.mask[i, 0, 1, 2] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == 'conv1x1-sobely':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 0, 1] = 2.0
                self.mask[i, 0, 0, 2] = 1.0
                self.mask[i, 0, 2, 0] = -1.0
                self.mask[i, 0, 2, 1] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == 'conv1x1-laplacian':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 1] = 1.0
                self.mask[i, 0, 1, 0] = 1.0
                self.mask[i, 0, 1, 2] = 1.0
                self.mask[i, 0, 2, 1] = 1.0
                self.mask[i, 0, 1, 1] = -4.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        else:
            raise ValueError('the type of seqconv is not supported!')

    def forward(self, x):
        y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
        # explicitly padding with bias
        y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
        b0_pad = self.b0.view(1, -1, 1, 1)
        y0[:, :, 0:1, :] = b0_pad
        y0[:, :, -1:, :] = b0_pad
        y0[:, :, :, 0:1] = b0_pad
        y0[:, :, :, -1:] = b0_pad
        # conv-3x3
        y1 = F.conv2d(input=y0, weight=self.scale * self.mask, bias=self.bias, stride=1, groups=self.out_planes)
        return y1

    def rep_params(self):
        device = self.k0.get_device()
        if device < 0:
            device = None
        tmp = self.scale * self.mask
        k1 = torch.zeros((self.out_planes, self.out_planes, 3, 3), device=device)
        for i in range(self.out_planes):
            k1[i, i, :, :] = tmp[i, 0, :, :]
        b1 = self.bias
        # re-param conv kernel
        RK = F.conv2d(input=k1, weight=self.k0.permute(1, 0, 2, 3))
        # re-param conv bias
        RB = torch.ones(1, self.out_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
        RB = F.conv2d(input=RB, weight=k1).view(-1, ) + b1
        return RK, RB


class RepBlock(nn.Module):
    def __init__(self, in_channels, out_channels, deploy=False):
        super(RepBlock, self).__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation('lrelu')

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(3, 3), stride=1,
                                         padding=1, dilation=1, groups=1, bias=True,
                                         padding_mode='zeros')
        else:
            self.rbr_3x3_branch = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3),
                                            stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros')

            self.rbr_conv1x1_sbx_branch = SeqConv3x3('conv1x1-sobelx', self.in_channels, self.out_channels)
            self.rbr_conv1x1_sby_branch = SeqConv3x3('conv1x1-sobely', self.in_channels, self.out_channels)
            self.rbr_conv1x1_lpl_branch = SeqConv3x3('conv1x1-laplacian', self.in_channels, self.out_channels)

    def forward(self, inputs):
        if (self.deploy):
            return self.activation(self.rbr_reparam(inputs))
        else:
            return self.activation(
                self.rbr_3x3_branch(inputs) + inputs + self.rbr_conv1x1_sbx_branch(
                    inputs) + self.rbr_conv1x1_sby_branch(inputs) + self.rbr_conv1x1_lpl_branch(inputs))


class RepRFB(nn.Module):
    def __init__(self, feature_nums, out_chs, act_type='lrelu', deploy=False):
        super(RepRFB, self).__init__()
        self.repblock1 = RepBlock(in_channels=feature_nums, out_channels=feature_nums, deploy=deploy)
        self.repblock2 = RepBlock(in_channels=feature_nums, out_channels=feature_nums, deploy=deploy)
        self.repblock3 = RepBlock(in_channels=feature_nums, out_channels=feature_nums, deploy=deploy)

        self.conv3 = nn.Conv2d(in_channels=feature_nums, out_channels=out_chs, kernel_size=3, stride=1, padding=1)

        self.act = activation('lrelu')

    def forward(self, inputs):
        outputs = self.repblock1(inputs)
        outputs = self.repblock2(outputs)
        outputs = self.repblock3(outputs)
        outputs = inputs + outputs
        outputs = self.act(self.conv3(outputs))

        return outputs


class conv(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)
        # self.bn   = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # print(x.size())
        return F.leaky_relu(self.conv(x), negative_slope=0.2)




class FusionNet_Splitv2(nn.Module):
    def __init__(self):
        super(FusionNet_Splitv2, self).__init__()
        ch = [32, 64, 128, 256]
        ch_seg = [128, 320]
        self.conv1 = conv(2, ch[0])
        self.repblock1 = RepRFB(ch[0], ch[1])

        self.repblock2 = RepRFB(ch[1], ch[2])
        self.repblock3 = RepRFB(ch[2], ch[3])

        self.en_conv1 = Enhance_module(ch[2]//2, ch_seg[0])

        self.sim1 = SIM_v2(ch_seg[0], ch[2]//2)

        self.en_conv2 = Enhance_module(ch[3]//2, ch_seg[1])
        self.sim2 = SIM_v2(ch_seg[1], ch[3]//2)

        self.rec_conv1 = conv(ch[3], ch[2])
        self.rec_conv2 = conv(ch[2], ch[1])
        self.rec_conv3 = conv(ch[1], ch[0])
        self.rec_conv4 = nn.Sequential(
                nn.Conv2d(ch[0], 1, (3, 3), (1, 1), 1),
                nn.Tanh()
            )

    def forward(self, vi, ir):
        feat_list = []
        f_shallow = self.conv1(torch.cat((ir, vi), dim=1))

        f1 = self.repblock1(f_shallow)


        f2 = self.repblock2(f1)
        f2_1, f2_2 = f2.chunk(2, dim=1)
        feat1 = self.en_conv1(f2_1)
        feat_list.append(feat1)
        f_sim1 = self.sim1(feat1, f2_1)
        f_deep1 = torch.cat((f_sim1, f2_2), dim=1)

        f3 = self.repblock3(f_deep1)
        f3_1, f3_2 = f3.chunk(2, dim=1)
        feat2 = self.en_conv2(f3_1)
        feat_list.append(feat2)
        f_sim2 = self.sim2(feat2, f3_1)
        f_deep2 = torch.cat((f_sim2, f3_2), dim=1)

        d1 = self.rec_conv1(f_deep2) + f_deep1
        d2 = self.rec_conv2(d1) + f1
        d3 = self.rec_conv3(d2) + f_shallow

        output = self.rec_conv4(d3)

        return feat_list, output


class FusionNet_visual(nn.Module):
    def __init__(self):
        super(FusionNet_visual, self).__init__()
        ch = [32, 64, 128, 256]
        ch_seg = [128, 320]
        self.conv1 = conv(2, ch[0])
        self.repblock1 = RepRFB(ch[0], ch[1])

        self.repblock2 = RepRFB(ch[1], ch[2])
        self.repblock3 = RepRFB(ch[2], ch[3])

        self.en_conv1 = Enhance_module(ch[2]//2, ch_seg[0])

        self.sim1 = SIM_v2(ch_seg[0], ch[2]//2)

        self.en_conv2 = Enhance_module(ch[3]//2, ch_seg[1])
        self.sim2 = SIM_v2(ch_seg[1], ch[3]//2)

        self.rec_conv1 = conv(ch[3], ch[2])
        self.rec_conv2 = conv(ch[2], ch[1])
        self.rec_conv3 = conv(ch[1], ch[0])
        self.rec_conv4 = nn.Sequential(
                nn.Conv2d(ch[0], 1, (3, 3), (1, 1), 1),
                nn.Tanh()
            )

    def forward(self, vi, ir):
        feat_list = []
        f_shallow = self.conv1(torch.cat((ir, vi), dim=1))

        f1 = self.repblock1(f_shallow)


        f2 = self.repblock2(f1)
        f2_1, f2_2 = f2.chunk(2, dim=1)
        feat1 = self.en_conv1(f2_1)
        feat_list.append(feat1)
        f_sim1 = self.sim1(feat1, f2_1)
        f_deep1 = torch.cat((f_sim1, f2_2), dim=1)

        f3 = self.repblock3(f_deep1)
        f3_1, f3_2 = f3.chunk(2, dim=1)
        feat2 = self.en_conv2(f3_1)
        feat_list.append(feat2)
        f_sim2 = self.sim2(feat2, f3_1)
        f_deep2 = torch.cat((f_sim2, f3_2), dim=1)

        d1 = self.rec_conv1(f_deep2) + f_deep1
        d2 = self.rec_conv2(d1) + f1
        d3 = self.rec_conv3(d2) + f_shallow

        output = self.rec_conv4(d3)

        return f1 ,feat1, f_sim1, output


class noSDIM(nn.Module):
    def __init__(self):
        super(noSDIM, self).__init__()
        ch = [32, 64, 128, 256]
        ch_seg = [128, 320]
        self.conv1 = conv(2, ch[0])
        self.repblock1 = RepRFB(ch[0], ch[1])

        self.repblock2 = RepRFB(ch[1], ch[2])
        self.repblock3 = RepRFB(ch[2], ch[3])

        self.en_conv1 = Enhance_module(ch[2]//2, ch_seg[0])
        self.conv_en1 = conv(ch_seg[0], ch[2]//2)
        
        # self.sim1 = SIM_v2(ch_seg[0], ch[2])

        self.en_conv2 = Enhance_module(ch[3]//2, ch_seg[1])
        self.conv_en2 = conv(ch_seg[1], ch[3]//2)
        # self.sim2 = SIM_v2(ch_seg[1], ch[3])

        self.rec_conv1 = conv(ch[3], ch[2])
        self.rec_conv2 = conv(ch[2], ch[1])
        self.rec_conv3 = conv(ch[1], ch[0])
        self.rec_conv4 = nn.Sequential(
                nn.Conv2d(ch[0], 1, (3, 3), (1, 1), 1),
                nn.Tanh()
            )

    def forward(self, vi, ir):
        feat_list = []
        f_shallow = self.conv1(torch.cat((ir, vi), dim=1))

        f1 = self.repblock1(f_shallow)


        f2 = self.repblock2(f1)
        f2_1, f2_2 = f2.chunk(2, dim=1)
        feat1 = self.en_conv1(f2_1)
        fs1 = self.conv_en1(feat1)
        feat_list.append(feat1)
        # f_sim1 = self.sim1(feat1, f2_1)
        f_deep1 = torch.cat((fs1, f2_2), dim=1)

        f3 = self.repblock3(f_deep1)
        f3_1, f3_2 = f3.chunk(2, dim=1)
        feat2 = self.en_conv2(f3_1)
        fs2 = self.conv_en2(feat2)
        f_deep2 = torch.cat((fs2, f3_2), dim=1)
        feat_list.append(feat2)

        d1 = self.rec_conv1(f_deep2) + f_deep1
        d2 = self.rec_conv2(d1) + f1
        d3 = self.rec_conv3(d2) + f_shallow

        output = self.rec_conv4(d3)

        return feat_list, output

class directInject(nn.Module):
    def __init__(self):
        super(directInject, self).__init__()
        ch = [32, 64, 128, 256]
        ch_seg = [128, 320]
        self.conv1 = conv(2, ch[0])
        self.repblock1 = RepRFB(ch[0], ch[1])

        self.repblock2 = RepRFB(ch[1], ch[2])
        self.repblock3 = RepRFB(ch[2], ch[3])

        # self.en_conv1 = Enhance_module(ch[2], ch_seg[0])

        self.sim1 = SIM_v2(ch_seg[0], ch[2]//2)

        # self.en_conv2 = Enhance_module(ch[3], ch_seg[1])
        self.sim2 = SIM_v2(ch_seg[1], ch[3]//2)
        self.up1 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.up2 = nn.Upsample(scale_factor=16, mode='bilinear')
        # self.up1 = nn.ConvTranspose2d(128, 128, kernel_size=16, stride=8, padding=4, output_padding=0)
        # self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=32, stride=16, padding=8, output_padding=0)
    
        self.rec_conv1 = conv(ch[3], ch[2])
        self.rec_conv2 = conv(ch[2], ch[1])
        self.rec_conv3 = conv(ch[1], ch[0])
        self.rec_conv4 = nn.Sequential(
                nn.Conv2d(ch[0], 1, (3, 3), (1, 1), 1),
                nn.Tanh()
            )

    def forward(self, vi, ir,seg1,seg2):
        
        f_shallow = self.conv1(torch.cat((ir, vi), dim=1))

        f1 = self.repblock1(f_shallow)


        f2 = self.repblock2(f1)
        f2_1, f2_2 = f2.chunk(2, dim=1)
        seg1 = self.up1(seg1)
        seg2 = self.up2(seg2)
        f_sim1 = self.sim1(seg1, f2_1)
        f_deep1 = torch.cat((f_sim1, f2_2), dim=1)

        f3 = self.repblock3(f_deep1)
        f3_1,f3_2 = f3.chunk(2,dim=1)
        f_sim2 = self.sim2(seg2, f3_1)
        f_deep2 = torch.cat((f_sim2, f3_2), dim=1)

        d1 = self.rec_conv1(f_deep2)+f_deep1
        d2 = self.rec_conv2(d1)+f1
        d3 = self.rec_conv3(d2)+f_shallow

        output = self.rec_conv4(d3)

        return output

class noMIDM(nn.Module):
    def __init__(self):
        super(noMIDM, self).__init__()
        ch = [32, 64, 128, 256]
        ch_seg = [128, 320]
        self.conv1 = conv(2, ch[0])
        self.repblock1 = RepRFB(ch[0], ch[1])

        self.repblock2 = RepRFB(ch[1], ch[2])
        self.repblock3 = RepRFB(ch[2], ch[3])
        self.en1 = nn.Sequential(
                nn.Conv2d(in_channels=ch[2], out_channels=ch_seg[0],kernel_size=1,stride=1),
                nn.BatchNorm2d(ch_seg[0]),
                nn.ReLU(inplace=True),
                conv(ch_seg[0],ch_seg[0])

        )
        self.en2 = nn.Sequential(
                nn.Conv2d(in_channels=ch[3], out_channels=ch_seg[1],kernel_size=1,stride=1),
                nn.BatchNorm2d(ch_seg[1]),
                nn.ReLU(inplace=True),
                conv(ch_seg[1],ch_seg[1])
        )

        self.en_conv1 = Enhance_module(ch[2]//2, ch_seg[0])

        self.sim1 = SIM_v2(ch_seg[0], ch[2]//2)

        self.en_conv2 = Enhance_module(ch[3]//2, ch_seg[1])
        self.sim2 = SIM_v2(ch_seg[1], ch[3]//2)

        self.rec_conv1 = conv(ch[3], ch[2])
        self.rec_conv2 = conv(ch[2], ch[1])
        self.rec_conv3 = conv(ch[1], ch[0])
        self.rec_conv4 = nn.Sequential(
                nn.Conv2d(ch[0], 1, (3, 3), (1, 1), 1),
                nn.Tanh()
            )

    def forward(self, vi, ir):
        feat_list = []
        f_shallow = self.conv1(torch.cat((ir, vi), dim=1))

        f1 = self.repblock1(f_shallow)


        f2 = self.repblock2(f1)
        feat1 = self.en1(f2)
        feat_list.append(feat1)

        f3 = self.repblock3(f2)
        feat2 = self.en2(f3)
        feat_list.append(feat2)

        d1 = self.rec_conv1(f3) + f2
        d2 = self.rec_conv2(d1) + f1
        d3 = self.rec_conv3(d2) + f_shallow

        output = self.rec_conv4(d3)

        return feat_list, output



class noMSPM(nn.Module):
    def __init__(self):
        super(noMSPM, self).__init__()
        ch = [32, 64, 128, 256]
        ch_seg = [128, 320]
        self.conv1 = conv(2, ch[0])
        self.repblock1 = RepRFB(ch[0], ch[1])

        self.repblock2 = RepRFB(ch[1], ch[2])
        self.repblock3 = RepRFB(ch[2], ch[3])

        self.en_conv1 = nn.Sequential(
                nn.Conv2d(in_channels=ch[2]//2, out_channels=ch_seg[0],kernel_size=1,stride=1),
                conv(ch_seg[0],ch_seg[0])

        )
            
        
        self.sim1 = SIM_v2(ch_seg[0], ch[2]//2)

        self.en_conv2 = nn.Sequential(
                nn.Conv2d(in_channels=ch[3]//2, out_channels=ch_seg[1],kernel_size=1,stride=1),
                conv(ch_seg[1],ch_seg[1])
        )
        self.sim2 = SIM_v2(ch_seg[1], ch[3]//2)

        self.rec_conv1 = conv(ch[3], ch[2])
        self.rec_conv2 = conv(ch[2], ch[1])
        self.rec_conv3 = conv(ch[1], ch[0])
        self.rec_conv4 = nn.Sequential(
                nn.Conv2d(ch[0], 1, (3, 3), (1, 1), 1),
                nn.Tanh()
            )
    
    def forward(self, vi, ir):
        feat_list = []
        f_shallow = self.conv1(torch.cat((ir, vi), dim=1))

        f1 = self.repblock1(f_shallow)


        f2 = self.repblock2(f1)
        f2_1, f2_2 = f2.chunk(2, dim=1)
        feat1 = self.en_conv1(f2_1)
        feat_list.append(feat1)
        f_sim1 = self.sim1(feat1, f2_1)
        f_deep1 = torch.cat((f_sim1, f2_2), dim=1)

        f3 = self.repblock3(f_deep1)
        f3_1, f3_2 = f3.chunk(2, dim=1)
        feat2 = self.en_conv2(f3_1)
        feat_list.append(feat2)
        f_sim2 = self.sim2(feat2, f3_1)
        f_deep2 = torch.cat((f_sim2, f3_2), dim=1)

        d1 = self.rec_conv1(f_deep2) + f_deep1
        d2 = self.rec_conv2(d1) + f1
        d3 = self.rec_conv3(d2) + f_shallow

        output = self.rec_conv4(d3)

        return feat_list, output




