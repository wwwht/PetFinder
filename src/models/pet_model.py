'''
Author: your name
Date: 2021-10-31 16:24:15
LastEditTime: 2021-11-03 21:45:12
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \PetFinder\src\models\pet_model.py
'''
import sys
# sys.path.append("..")
from typing import Any, List  
import torch 
from pytorch_lightning import LightningModule
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from torchmetrics.classification.accuracy import Accuracy
from src.models.modules.simple_dense_net import SimpleDenseNet
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
# from datamodules.transformer.default_tranform import default_transforms

def mixup(x: torch.Tensor, y:torch.Tensor, alpha:float = 1.0):
    assert alpha > 0, "alpha should be larger than 0"
    assert x.size(0) > 1, "Mixup cannot be applied to a single instance."

    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0])
    mixed_x = lam * x + (1 - lam) * x[rand_index, :]
    target_a, target_b = y, y[rand_index]
    return mixed_x, target_a, target_b, lam

class PetModel(LightningModule):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.cfg = cfg
        self.__build_model()
        self._criterion = eval(self.cfg.loss)()
        self.save_hyperparameters()
        # self.transform = default_transforms(cfg.image_size)

    def __build_model(self):
        self.backbone = create_model(
            self.cfg.model.name, pretrained=True, num_classes=0, in_chans=3
        )
        num_features = self.backbone.num_features
        self.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(num_features, self.cfg.model.output_dim)
        )

    def forward(self, x: torch.Tensor):
        # return self.model(x)
        f = self.backbone(x)
        out = self.fc(f)
        return out

    def step(self, batch: Any, mode: str):
        images, labels = batch
        labels = labels.float() / 100.0
        # images = self.transform[mode](images)
        
        if torch.rand(1)[0] < 0.5 and mode == 'train':
            mix_images, target_a, target_b, lam = mixup(images, labels, alpha=0.5)
            logits = self.forward(mix_images).squeeze(1)
            loss = self._criterion(logits, target_a) * lam + \
                (1 - lam) * self._criterion(logits, target_b)
        else:
            logits = self.forward(images).squeeze(1)
            loss = self._criterion(logits, labels)
        
        pred = logits.sigmoid().detach().cpu() * 100.
        labels = labels.detach().cpu() * 100.
        return loss, pred, labels

    def training_step(self, batch: Any, batch_idx: int):
        loss, pred, targets = self.step(batch, 'train')

        # log train metrics
        # acc = self.train_accuracy(preds, targets)
        # self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        # self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "pred": pred, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        self.__share_epoch_end(outputs, 'train')

    def validation_step(self, batch: Any, batch_idx: int):
        loss, pred, labels = self.step(batch, 'val')
        return {'pred': pred, 'labels': labels}

    def validation_epoch_end(self, outputs: List[Any]):
        self.__share_epoch_end(outputs, 'val')

    def __share_epoch_end(self, outputs, mode):
        preds = []
        labels = []
        for out in outputs:
            pred, label = out['pred'], out['labels']
            preds.append(pred)
            labels.append(label)
        preds = torch.cat(preds)
        labels = torch.cat(labels)
        metrics = torch.sqrt(((labels - preds) ** 2).mean())
        self.log(f'{mode}_loss', metrics)

    def check_gradcam(self, dataloader, target_layer, target_category, reshape_transform=None):
        cam = GradCAMPlusPlus(
            model=self,
            target_layer=target_layer, 
            use_cuda=self.cfg.trainer.gpus, 
            reshape_transform=reshape_transform)
        
        org_images, labels = iter(dataloader).next()
        cam.batch_size = len(org_images)
        images = self.transform['val'](org_images)
        images = images.to(self.device)
        logits = self.forward(images).squeeze(1)
        pred = logits.sigmoid().detach().cpu().numpy() * 100
        labels = labels.cpu().numpy()
        
        grayscale_cam = cam(input_tensor=images, target_category=target_category, eigen_smooth=True)
        org_images = org_images.detach().cpu().numpy().transpose(0, 2, 3, 1) / 255.
        return org_images, grayscale_cam, pred, labels
    

    # def configure_optimizers(self):
    #     """Choose what optimizers and learning-rate schedulers to use in your optimization.
    #     Normally you'd need one. But in the case of GANs or similar you might have multiple.

    #     See examples here:
    #         https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
    #     """
    #     return torch.optim.Adam(
    #         params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
    #     )
    def configure_optimizers(self):
        optimizer = eval(self.cfg.optimizer.name)(
            self.parameters(), **self.cfg.optimizer.params
        )
        scheduler = eval(self.cfg.scheduler.name)(
            optimizer,
            **self.cfg.scheduler.params
        )
        return [optimizer], [scheduler]