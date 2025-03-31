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
        # data_dir_vis = 'datasets/train/Visible'
        # data_dir_ir = 'datasets/train/Infrared'
        # data_dir_label = 'datasets/train/Label'
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
        data_dir_vis = '/image/vi' 
        data_dir_ir = '/image/ir'
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
    

class directInject_dataset(Dataset):
    def __init__(self, transform = to_tensor):
        super(directInject_dataset, self).__init__()
        data_dir_vis = '/mnt/raid1/data_work/MSRS/train/vi/'
        data_dir_ir = '/mnt/raid1/data_work/MSRS/train/ir/'
        data_dir_fuse = '/workspace/ycj22/PSFusion/Fusion_results/MSRS'
        # data_dir_fuse = 'Results/PSFusion'
        # data_dir_vis = 'datasets/train/Visible'
        # data_dir_ir = 'datasets/train/Infrared'
        # data_dir_label = 'datasets/train/Label'
        self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
        self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
        self.filepath_fuse, self.filenames_fuse = prepare_data_path(data_dir_fuse)
        self.length = min(len(self.filenames_vis), len(self.filenames_ir))
        self.transform  = transform
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        vis_path = self.filepath_vis[index]
        ir_path = self.filepath_ir[index]
        fuse_path = self.filepath_fuse[index]
        # label_path = self.filepath_label[index]
        image_vis = np.asarray(Image.open(vis_path), dtype=np.float32)/255.0
        image_fuse = np.asarray(Image.open(fuse_path), dtype=np.float32)/255.0
        image_inf = cv2.imread(ir_path, 0)
        image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
        # label = np.asarray(Image.open(label_path), dtype=np.int64)
        image_vis = self.transform(image_vis)
        image_ir = self.transform(image_ir)
        image_fuse = self.transform(image_fuse)
        # label = self.transform(label)
        name = self.filenames_vis[index]

        C, H, W = image_vis.shape
        y = random.randint(0,H-p-1)
        x = random.randint(0,W-p-1)
        image_ir = image_ir[:,y:y+p,x:x+p]
        image_vis = image_vis[:,y:y + p, x:x + p]
        image_fuse = image_fuse[:,y:y + p, x:x + p]
        # label = label[:, y:y + p, x:x + p]
        return image_vis, image_ir, image_fuse, name
    

class TestDI_dataset(Dataset):
    def __init__(self, transform = to_tensor):
        super(TestDI_dataset, self).__init__()
        # data_dir_vis = 'test_imgs/vi'
        # data_dir_ir = 'test_imgs/ir'
        # data_dir_vis = 'datasets/TNO/vi'
        # data_dir_ir = 'datasets/TNO/ir'
        # data_dir_vis = 'datasets/RoadScene/crop_HR_visible'
        # data_dir_ir = 'datasets/RoadScene/cropinfrared'
        data_dir_vis = '/mnt/raid1/data_work/MSRS/test/vi/'
        data_dir_ir = '/mnt/raid1/data_work/MSRS/test/ir/'

        data_dir_fuse = '/mnt/raid1/ycj22/融合结果/MSRS/PSFusion/'
        # data_dir_fuse ='/workspace/ycj22/Semantic_Fusion/Results/MSRS5102'

        # data_dir_fuse ='/workspace/ycj22/PSFusion/Fusion_results/MSRS'

        # data_dir_vis = 'datasets/test/Visible'
        # data_dir_ir = 'datasets/test/Infrared'
    
        self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
        self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
        self.filepath_fuse, self.filenames_fuse = prepare_data_path(data_dir_fuse)
        # self.filepath_label, self.filenames_label = prepare_data_path(data_dir_label)
        self.length = min(len(self.filenames_vis), len(self.filenames_ir))
        self.transform  = transform

    def __getitem__(self, index):
        vis_path = self.filepath_vis[index]
        ir_path = self.filepath_ir[index]
        fuse_path = self.filepath_fuse[index]
        # label_path = self.filepath_label[index]
        image = Image.open(vis_path)
        width,height = image.size
        image_vis = np.asarray(Image.open(vis_path), dtype=np.float32)/255.0
        image_fuse = np.asarray(Image.open(fuse_path), dtype=np.float32)/255.0
        image_inf = cv2.imread(ir_path, 0)
        image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
        # label = np.asarray(Image.open(label_path), dtype=np.int64)
        image_vis = self.transform(image_vis)
        image_ir = self.transform(image_ir)
        image_fuse = self.transform(image_fuse)
        # label = self.transform(label)
        name = self.filenames_vis[index]


        return image_vis, image_ir, image_fuse, name

    def __len__(self):
        return self.length


if __name__ == '__main__':
    train_dataset = directInject_dataset()
    print("the training dataset is length:{}".format(train_dataset.length))
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    train_loader.n_iter = len(train_loader)
    a = []
    for it, (image_vis, image_ir, image_fuse,name) in enumerate(train_loader):
        print(image_vis.size())
        print(image_ir.size())
        print(image_fuse.size())
        break
