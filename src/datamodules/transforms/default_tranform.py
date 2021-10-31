'''
Author: wwwht
Date: 2021-10-31 15:40:10
LastEditTime: 2021-10-31 15:53:11
LastEditors: Please set LastEditors
Description: 默认的数据增强方法
FilePath: \PetFinder\configs\datamodule\default_tranform.py
'''
import torchvision.transforms as T
import torch

IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB
IMAGENET_STD = [0.229, 0.224, 0.225]  # RGB

def default_transforms(image_size):
    transform = {
        "train": T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            T.Resize([image_size, image_size]),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
        "val": T.Compose([
            T.Resize([image_size, image_size]),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    }
    return transform