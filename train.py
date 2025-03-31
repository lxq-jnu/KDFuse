import os
 
import argparse
import torch
from model_seg import Network3
from model_fu import FusionNet_Splitv2
from torch.utils.data import DataLoader
from Datasets import Fusion_dataset
from loss import Fusionloss
from model_distiller import DistillerNet
import logging
from logger import setup_logger
import datetime
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(20)


def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--tchannel', type=list, default=[64, 128, 320, 512], help='teacher channel')
    parser.add_argument('--schannel', type=list, default=[16, 64, 128, 256], help='student channel')
    parser.add_argument('--batch_size', type=int, default=22, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=30, help='number of training epochs')
    parser.add_argument('--lr_start', type=float, default=0.001, help='learning rate')
    parser.add_argument('--modelpth', type=str, default='./models', help='saving model path')
    parser.add_argument('--logpath', type=str, default='./logs', help='log path')
 
    opt = parser.parse_args()

    if not os.path.isdir(opt.modelpth):
        os.makedirs(opt.modelpth)
    if not os.path.isdir(opt.logpath):
        os.makedirs(opt.logpath)

    return opt


if __name__ == '__main__':
    opt = parse_option()
    Loss_list = [] 

    logger1 = logging.getLogger()
    # setup_logger(opt.logpath)

    fusionmodel = FusionNet_Splitv2()

    segmodel = Network3('mit_b3', num_classes=9)

    dmodel = DistillerNet()

    model_list = nn.ModuleList([])
    model_list.append(fusionmodel)
    model_list.append(dmodel)
    model_list.append(segmodel)
    

    w = [0.5, 0.5]
    optimizer1 = torch.optim.Adam(fusionmodel.parameters(), lr=opt.lr_start)

    criterion_fusion = Fusionloss()



    if torch.cuda.is_available():
        model_list.cuda()

    train_dataset = Fusion_dataset()
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    st = glob_st = time.time()

    logger1.info('Training Fusion Model start~')
    for epoch in range(opt.epochs):

        print('\n| epo #%s begin...' % epoch)
        lr = opt.lr_start
        # if epoch < opt.epochs // 2:
        #     lr = opt.lr_start
        # else:
        #     lr = opt.lr_start * (opt.epochs - epoch) / (opt.epochs - opt.epochs // 2)

        for param_group in optimizer1.param_groups:
            param_group['lr'] = lr
        # set modules as train()
        for module in model_list:
            module.train()
        # set teacher as eval()
        model_list[-1].eval()

        model_s = model_list[0]
        model_t = model_list[-1]


        total_loss_epoch = 0
        ototal_loss_epoch = 0

        for it, (image_vis, image_ir, label, name) in enumerate(train_loader):
            image_vis = image_vis.cuda()
            image_ir = image_ir.cuda()
            label = label.cuda()
            image_vis_ycrcb = RGB2YCrCb(image_vis)
            image_vis_y = image_vis_ycrcb[:, :1]
            feat, out = model_s(image_vis_y, image_ir)
            fusion_ycrcb = torch.cat(
                (out, image_vis_ycrcb[:, 1:2, :, :],
                 image_vis_ycrcb[:, 2:, :, :]),
                dim=1,
            )

            fusion_image = YCrCb2RGB(fusion_ycrcb)

            optimizer1.zero_grad()
            fs, seg_map = model_t(fusion_image)

            dloss = dmodel(feat, fs)
            loss_fusion, loss_in, loss_grad = criterion_fusion(image_vis_ycrcb, image_ir, out)
            total_dloss = 0
            for i in range(len(dloss)):
                total_dloss += w[i] * dloss[i]
            


            loss_total = loss_fusion + 0.5 * total_dloss
            total_loss_epoch += loss_total.item()

            loss_total.backward()
            optimizer1.step()


            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            total_dloss = total_dloss.item()

            msg1 = ','.join(
                [
                    'epoch:{epoch}',
                    'pic_num:{it}',
                    'loss_total:{loss_total:.4f}',
                    'loss_in:{loss_in:.4f}',
                    'loss_grad:{loss_grad:.4f}',
                    'total_dloss:{total_dloss:.4f}',
                    'time: {time:.4f}',
                ]
            ).format(
                epoch=epoch,
                it=it,
                loss_total=loss_total.item(),
                loss_in=loss_in.item(),
                loss_grad=loss_grad.item(),
                # loss_fft=loss_fft.item(),
                total_dloss=total_dloss,
                time=t_intv,
            )

            logger1.info(msg1)

            st = ed
        Loss_list.append(total_loss_epoch / (it + 1))

        if (epoch + 1) % 30 == 0:
                torch.save(fusionmodel.state_dict(), f'{opt.modelpth}/fusion_model_SIM_epoch_{epoch + 1}.pth')
                logger1.info("Fusion Model Save to: {}".format(f'{opt.modelpth}/fusion_model_SIM_epoch_{epoch + 1}.pth'))
