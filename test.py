import os

from model_seg import Network3
import torch
from torch.utils.data import DataLoader
from Datasets import Test_dataset,TestDI_dataset
from model_fu import FusionNet_Splitv2,directInject
from tqdm import tqdm
from skimage import img_as_ubyte
from skimage.io import imsave
import numpy as np
from PIL import Image
import time
def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)  # (nhw,c)
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


def test(fusion_model_path='/models/fusion_model_SIM_epoch_30.pth'):
    # save_dir = './fusion_result/v2new_sa'
    save_dir = './Results/KDFuse'

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # fusionmodel = directKD()
    fusionmodel = FusionNet_Splitv2()
    # fusionmodel = onlyfusion()
    # fusionmodel = nosim()

    # fusionmodel = onlyVREM()
    # fusionmodel = noms2p()
    fusionmodel.cuda()
    fusionmodel.load_state_dict(torch.load(fusion_model_path))

    print('model load done')
    test_dataset = Test_dataset()
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    test_bar = tqdm(test_loader)

    total_time = 0.0
    num_images = len(test_dataset)

    with torch.no_grad():
        for it, (img_vis, img_ir, name) in enumerate(test_bar):
            img_vis = img_vis.cuda()
            img_ir = img_ir.cuda()
            img_vis_ycrcb = RGB2YCrCb(img_vis)
            image_vis_y = img_vis_ycrcb[:, :1]

            start_time = time.time()

            _, out= fusionmodel(image_vis_y, img_ir)
            
            end_time = time.time()

            elapsed_time = end_time - start_time
            total_time += elapsed_time
            fusion_ycrcb = torch.cat(
                (out, img_vis_ycrcb[:, 1:2, :, :],
                 img_vis_ycrcb[:, 2:, :, :]),
                dim=1,
            )


            fusion_image = YCrCb2RGB(fusion_ycrcb)

            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)
            fused_image = fusion_image.cpu().numpy()
            fused_image = np.uint8(255.0 * fused_image)

            fused_image = fused_image.transpose((0, 2, 3, 1))
            fused_image = (fused_image - np.min(fused_image)) / (
                    np.max(fused_image) - np.min(fused_image)
            )

            fused_image = np.uint8(255.0 * fused_image)

            

            for k in range(len(name)):
                img_name = name[k]
                image = fused_image[k, :, :, :]
                save_path = os.path.join(save_dir, img_name)
                image = Image.fromarray(image)
                image.save(save_path)
                test_bar.set_description('Fusion {0} Sucessfully!'.format(name[k]))
    avg_time_per_image = total_time / num_images
    print("Average processing time per image: {:.4f}s".format(avg_time_per_image))

def test_dI(fusion_model_path='/workspace/ycj22/Semantic_Fusion/model/directinject_fnopre/directInject30.pth'):
    # save_dir = './fusion_result/v2new_sa'
    save_dir = './Results/directInject_fnopre'


    segmodel = Network3('mit_b3', num_classes=9)
    segmodel = segmodel.cuda()
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    fusionmodel = directInject()
    # fusionmodel = onlyfusion()
    # fusionmodel = nosim()
    fusionmodel.cuda()
    fusionmodel.load_state_dict(torch.load(fusion_model_path))

    print('model load done')
    test_dataset = TestDI_dataset()
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    test_bar = tqdm(test_loader)
    with torch.no_grad():
        for it, (img_vis, img_ir, img_fuse, name) in enumerate(test_bar):
            img_vis = img_vis.cuda()
            img_ir = img_ir.cuda()
            img_fuse = img_fuse.cuda()
            img_vis_ycrcb = RGB2YCrCb(img_vis)
            image_vis_y = img_vis_ycrcb[:, :1]
            segmodel.eval()
            img_f = (img_vis+img_ir)/2
            img_f.cuda()
            fs, seg = segmodel(img_f)
            out= fusionmodel(image_vis_y, img_ir,fs[0],fs[1])
            fusion_ycrcb = torch.cat(
                (out, img_vis_ycrcb[:, 1:2, :, :],
                 img_vis_ycrcb[:, 2:, :, :]),
                dim=1,
            )


            fusion_image = YCrCb2RGB(fusion_ycrcb)

            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)
            fused_image = fusion_image.cpu().numpy()
            fused_image = np.uint8(255.0 * fused_image)

            fused_image = fused_image.transpose((0, 2, 3, 1))
            fused_image = (fused_image - np.min(fused_image)) / (
                    np.max(fused_image) - np.min(fused_image)
            )

            fused_image = np.uint8(255.0 * fused_image)

            

            for k in range(len(name)):
                img_name = name[k]
                image = fused_image[k, :, :, :]
                save_path = os.path.join(save_dir, img_name)
                image = Image.fromarray(image)
                image.save(save_path)
                test_bar.set_description('Fusion {0} Sucessfully!'.format(name[k]))


if __name__ =='__main__':
    test()


