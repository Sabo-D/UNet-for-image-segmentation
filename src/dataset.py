import os
from tkinter import image_names

import torch
import numpy as np
from torch.backends.cudnn import set_flags
from torch.nn.functional import bilinear
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from torchvision.transforms import InterpolationMode

from src.test_model import model_test


class MyDataset(Dataset):
    """
    用于train, val, test
    """
    # root_dir: data目录
    def __init__(self, root_dir, new_size=(256, 256)):
        self.root_dir = root_dir
        self.new_size = new_size
        # 取得segments 后缀png
        self.segment = os.listdir(os.path.join(root_dir, 'Segmentation'))
        self.segment = [img for img in self.segment if img.endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.segment)

    def __getitem__(self, idx):
        """
        对数据采用PIL->ndarray->tensor
        因为segments中的数据默认是P模式打开 单通道
        每个像素值为类型索性 并非数值强度
        如果采用ToTensor直接PIL->tensor会做归一化
        导致类别信息破坏 crossEntropy要求类别信息完整
        :param idx:
        :return:
        """
        segment_name = self.segment[idx]
        # segment.png
        segment_path = os.path.join(self.root_dir, 'Segmentation', segment_name)
        # image.jpg
        image_path = os.path.join(self.root_dir, 'JPEGImages', segment_name.replace('.png', '.jpg'))
        # PIL
        image = Image.open(image_path).convert('RGB')
        segment = Image.open(segment_path)
        # resize
        image = image.resize(self.new_size, Image.Resampling.BILINEAR)
        segment = segment.resize(self.new_size, Image.Resampling.NEAREST)
        # ndarray （H,W,C） 注意如果是一通道(H,W) 并非（H,W,1）
        image = np.array(image, dtype=np.uint8)  # (H,W,3)
        segment = np.array(segment, dtype=np.uint8)  # (H,W)
        # tensor 调整类型 并调整顺序 （C,H,W）
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        # segment[segment == 255] = 0
        segment = torch.tensor(segment, dtype=torch.long)

        return image, segment

class InferenceDataset(Dataset):
    def __init__(self, root_dir, new_size=(256, 256), num=5000):
        self.root_dir = root_dir
        self.new_size = new_size
        self.num = num
        # 所有segments列表 带有后缀png
        self.segment = os.listdir(os.path.join(root_dir, 'Segmentation'))
        self.segment = [img for img in self.segment if img.endswith(('.jpg', '.jpeg', '.png'))]
        # 所有images列表 带有后缀jpg
        self.images = os.listdir(os.path.join(root_dir, 'JPEGImages'))
        # segments的base_names
        self.segment_names = [f.split('.')[0] for f in self.segment]
        # images的base_names
        self.image_names = [f.split('.')[0] for f in self.images]
        # 做差 得到没有标签的images列表 具有后缀jpg
        self.images_without_segment = [img for img in self.images if img.split('.')[0] not in self.segment_names]
        # 取定量
        self.images_without_segment = self.images_without_segment[:num]

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        image = self.images_without_segment[idx]
        image_name = image.split('.')[0]
        image_path = os.path.join(self.root_dir, 'JPEGImages', image)
        # PIL
        image = Image.open(image_path).convert('RGB')
        origin_size = image.size
        image = image.resize(self.new_size, Image.Resampling.BILINEAR)
        # ndarray
        image = np.array(image, dtype=np.uint8)
        # tensor
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        return image, image_name, origin_size

def inference_dataloader():
    """
    (B,C,H,W)
    :return:
    """
    root_dir = 'D:\AA_Pycharm_Projs\\UNet\data'
    dataset = InferenceDataset(root_dir, num=5000)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    return dataloader

def train_val_test_dataloader():
    """
    images是（B,C,H,W） segments是（B,H,W）
    :return:
    """
    root_dir ='D:\AA_Pycharm_Projs\\UNet01\data01'

    dataset = MyDataset(root_dir)
    train_size = round(0.75 * len(dataset))
    val_size = round(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset,[train_size, val_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader

if __name__ == '__main__':

    train_dataloader, val_dataloader, test_dataloader = train_val_test_dataloader()
    for images, segments in test_dataloader:

       print(images.shape)
       print(segments.shape)
       print(segments)
       print(torch.unique(segments))



