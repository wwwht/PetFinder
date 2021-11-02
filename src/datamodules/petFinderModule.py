'''
Author: your name
Date: 2021-10-31 07:44:35
LastEditTime: 2021-11-02 21:08:14
LastEditors: Please set LastEditors
Description: Pet Finder task data module
FilePath: \PetFinder\src\datamodules\petFinderModule.py
'''

from typing import Optional, Tuple
import pandas as pd
import numpy as np
import torch
import os
import glob
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
# from torchvision.transforms import T
from torchvision.io import read_image
from .transformer.default_tranform import default_transforms

class PetFinderDataset(Dataset):
    '''
    df: dataFrame
    transforms: 数据增强
    image_size: 输入网络的图像尺寸
    '''
    def __init__(self, df, transforms,image_size, mode):
        self._X = df['Id'].values
        self._y = None
        if "Pawpularity" in df.keys():
            self._y = df['Pawpularity'].values
        # self._transform = T.Resize([image_size, image_size])
        self._transform = transforms
        self._image_size = image_size
        self.mode = mode

    def __len__(self):
        return len(self._X)

    def __getitem__(self,idx):
        image_path = self._X[idx]
        image = read_image(image_path)
        image = self._transform[self.mode](image)
        if self._y is not None:
            label = self._y[idx]
            return image, label
        return image

class PetFinderDataModule(LightningDataModule):
    def __init__(
        self,
        # transform,
        root_path:str = "A:\Kaggle\PetFinder",
        batch_size:int = 64,
        num_workers:int = 0,
        pin_memory:bool = False,
        shuffle:bool = True,
        drop_last:bool = True,
        image_size:int = 224,
    ):
        super().__init__()
        # self._train_df = train_df
        # self._val_df = val_df
        self._root_path = root_path
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._image_size = image_size
        self._train_df = None
        self._val_df = None 
        self._transform = default_transforms(self._image_size)
        

    def setup(self, stage: Optional[str] = None):
        df = pd.read_csv(os.path.join(self._root_path, "train.csv"))
        df["Id"] = df["Id"].apply(lambda x: os.path.join(self._root_path, "train", x + ".jpg"))
        if not self._train_df or not self._val_df:
            self._train_df = df
            self._val_df = df

    def __create_dataset(self, train=True):
        return (
            PetFinderDataset(self._train_df, self._transform, self._image_size, "train")
            if train
            else PetFinderDataset(self._val_df, self._transform, self._image_size, "val")
        )
    def train_dataloader(self):
        dataset = self.__create_dataset(True)
        return DataLoader(dataset = dataset, 
            batch_size = self._batch_size,
            num_workers = self._num_workers,
            pin_memory = self._pin_memory,
            shuffle = True,
            drop_last = True)

    def val_dataloader(self):
        dataset = self.__create_dataset(False)
        return DataLoader(dataset = dataset,          
            batch_size = self._batch_size,         
            num_workers = self._num_workers,         
            pin_memory = self._pin_memory,         
            shuffle = False,         
            drop_last = False)
