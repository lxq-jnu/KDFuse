import torch
import torch.nn as nn
import torch.nn.functional as F
from segformer_head import SegFormerHead
import mix_transformer
from timm.models.layers import trunc_normal_
import math
from mmcv.cnn import ConvModule
class WeTr(nn.Module):
    def __init__(self, backbone, num_classes=9, embedding_dim=256, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.backbone = backbone
        self.feature_strides = [4, 8, 16, 32]
        # self.in_channels = [32, 64, 160, 256]
        # self.in_channels = [64, 128, 320, 512]

        self.encoder = getattr(mix_transformer,backbone)()  
        self.in_channels = self.encoder.embed_dims
        ## initilize encoder
        if pretrained:
            state_dict = torch.load('pretrained/' + backbone + '.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict, )

        self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels,
                                     embedding_dim=self.embedding_dim, num_classes=self.num_classes)

        self.classifier = nn.Conv2d(in_channels=self.in_channels[-1], out_channels=self.num_classes, kernel_size=1,
                                    bias=False)

    def initialize(self, ):
        state_dict = torch.load('pretrained/' + self.backbone + '.pth')
        state_dict.pop('head.weight')
        state_dict.pop('head.bias')
        self.encoder.load_state_dict(state_dict, )

    def _forward_cam(self, x):
        cam = F.conv2d(x, self.classifier.weight)
        cam = F.relu(cam)

        return cam

    def get_param_groups(self):

        param_groups = [[], [], []]  #

        for name, param in list(self.encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        for param in list(self.decoder.parameters()):
            param_groups[2].append(param)

        param_groups[2].append(self.classifier.weight)

        return param_groups

    def forward(self, x):

        _x = self.encoder(x)
        _x1, _x2, _x3, _x4 = _x
        cls = self.classifier(_x4)

        return _x, self.decoder(_x)




class Network3(nn.Module):

    def __init__(self,backbone, num_classes=9, embedding_dim=256): 
        super(Network3, self).__init__()

        self.fusion_nums = 2
        self.seg_nums = 2
        self.fusion_channel = 48
        self.seg_channel = 64
        self.denoise_net = WeTr(backbone,num_classes,embedding_dim)  #backbone=mit-b3
        # self.discriminator = Discriminator()
        self.mean =[123.675, 116.28, 103.53]
        self.std = [58.395, 57.12, 57.375]
    def forward(self, fused_seg1):

        torch_norma = fused_seg1*255
        for index in range(3):
            torch_norma[:,index,:,:] = (torch_norma[:,index,:,:] - self.mean[index])/self.std[index]

        feature_s,seg_map = self.denoise_net(torch_norma)
        f_s1,f_s2,f_s3,f_s4 = feature_s
        # return [f_s1,f_s2,f_s3,f_s4],seg_map
        return [f_s2, f_s3], seg_map

    def _loss(self,fused_seg1,label,criterion):
        torch_norma = fused_seg1 * 255
        for index in range(3):
            torch_norma[:, index, :, :] = (torch_norma[:, index, :, :] - self.mean[index]) / self.std[index]
        _, seg_map = self.denoise_net(torch_norma)
        outputs = F.interpolate(seg_map, size=label.shape[1:], mode='bilinear', align_corners=False)
        denoise_loss = criterion(outputs,label.type(torch.long))
        return denoise_loss


    def enhance_net_parameters(self):
        return self.enhance_net.parameters()

    def denoise_net_parameters(self):
        return self.denoise_net.parameters()

class Network3_dual(nn.Module):

    def __init__(self,backbone, num_classes=9, embedding_dim=256):
        super(Network3_dual, self).__init__()

        self.fusion_nums = 2
        self.seg_nums = 2
        self.fusion_channel = 48
        self.seg_channel = 64
        self.denoise_net = WeTr(backbone,num_classes,embedding_dim)  #backbone=mit-b3
        # self.discriminator = Discriminator()
        self.mean =[123.675, 116.28, 103.53]
        self.std = [58.395, 57.12, 57.375]
    def forward(self, fused_seg1):

        torch_norma = fused_seg1*255
        for index in range(3):
            torch_norma[:,index,:,:] = (torch_norma[:,index,:,:] - self.mean[index])/self.std[index]

        feature_s,seg_map = self.denoise_net(torch_norma)
        f_s1,f_s2,f_s3,f_s4 = feature_s
        # return [f_s1,f_s2,f_s3,f_s4],seg_map
        # return [f_s2, f_s3], seg_map
        return [f_s2], seg_map

    def _loss(self,fused_seg1,label,criterion):
        torch_norma = fused_seg1 * 255
        for index in range(3):
            torch_norma[:, index, :, :] = (torch_norma[:, index, :, :] - self.mean[index]) / self.std[index]
        _, seg_map = self.denoise_net(torch_norma)
        outputs = F.interpolate(seg_map, size=label.shape[1:], mode='bilinear', align_corners=False)
        denoise_loss = criterion(outputs,label.type(torch.long))
        return denoise_loss


    def enhance_net_parameters(self):
        return self.enhance_net.parameters()

    def denoise_net_parameters(self):
        return self.denoise_net.parameters()

