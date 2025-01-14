import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import cv2
import glob
import os
import torchvision.transforms as transforms
import random
to_tensor = transforms.Compose([transforms.ToTensor()])
p = 128

def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data.sort()
    filenames.sort()
    return data, filenames



class Fusion_dataset(Dataset):
    def __init__(self, transform = to_tensor):
        super(Fusion_dataset, self).__init__()
        data_dir_vis = '/mnt/raid1/data_work/MSRS/train/vi/'
        data_dir_ir = '/mnt/raid1/data_work/MSRS/train/ir/'
        data_dir_label = '/mnt/raid1/data_work/MSRS/train/mask/'
        self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
        self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
        self.filepath_label, self.filenames_label = prepare_data_path(data_dir_label)
        self.length = min(len(self.filenames_vis), len(self.filenames_ir))
        self.transform  = transform

    def __getitem__(self, index):
        vis_path = self.filepath_vis[index]
        ir_path = self.filepath_ir[index]
        label_path = self.filepath_label[index]
        image_vis = np.asarray(Image.open(vis_path), dtype=np.float32)/255.0
        image_inf = cv2.imread(ir_path, 0)
        image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
        label = np.asarray(Image.open(label_path), dtype=np.int64)
        image_vis = self.transform(image_vis)
        image_ir = self.transform(image_ir)
        label = self.transform(label)
        name = self.filenames_vis[index]

        C, H, W = image_vis.shape
        y = random.randint(0,H-p-1)
        x = random.randint(0,W-p-1)
        image_ir = image_ir[:,y:y+p,x:x+p]
        image_vis = image_vis[:,y:y + p, x:x + p]
        label = label[:, y:y + p, x:x + p]
        return image_vis, image_ir, label, name



    def __len__(self):
        return self.length
class Test_dataset(Dataset):
    def __init__(self, transform = to_tensor):
        super(Test_dataset, self).__init__()
        data_dir_vis = './image/vi/'
        data_dir_ir = './image/ir/'
        self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
        self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
        # self.filepath_label, self.filenames_label = prepare_data_path(data_dir_label)
        self.length = min(len(self.filenames_vis), len(self.filenames_ir))
        self.transform  = transform

    def __getitem__(self, index):
        vis_path = self.filepath_vis[index]
        ir_path = self.filepath_ir[index]
        # label_path = self.filepath_label[index]
        image = Image.open(vis_path)
        width,height = image.size
        image_vis = np.asarray(Image.open(vis_path), dtype=np.float32)/255.0
        image_inf = cv2.imread(ir_path, 0)
        image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
        # label = np.asarray(Image.open(label_path), dtype=np.int64)
        image_vis = self.transform(image_vis)
        image_ir = self.transform(image_ir)
        # label = self.transform(label)
        name = self.filenames_vis[index]

        return image_vis, image_ir, name

    def __len__(self):
        return self.length
