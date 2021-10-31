'''
Author: your name
Date: 2021-10-31 16:24:15
LastEditTime: 2021-10-31 16:30:36
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \PetFinder\src\models\pet_model.py
'''
from typing import Any, List  
import torch 
from pytorch_lightning import LightningModule
import numpy as np
import pandas as pd

def mixup(x: torch.Tensor, y:torch.Tensor, alpha:float = 1.0):
    assert alpha > 0, "alpha should be larger than 0"
    assert x.size(0) > 1, "Mixup cannot be applied to a single instance."

    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0])
    mixed_x = lam * x + (1 - lam) * x[rand_index, :]
    target_a, target_b = y, y[rand_index]
    return mixed_x, target_a, target_b, lam

class PetModel(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.__build