#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 18:37:43 2021

@author: lukas
"""

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm


class MixCropTimeSpecificModel(pl.LightningModule):
    def __init__(self, data_shape, use_model, num_targets, learning_rate, pretrained=True, activation=None, loss_func=None, target_transform=None, sigmoid_soft_factor=1.0, scheduler='one_cycle'):
        super().__init__()
        self.save_hyperparameters()
        self.target_transform=target_transform
        
        if use_model == 'res18':
            # model = torchvision.models.resnet18(pretrained=pretrained)
            # model.fc = nn.Linear(512, 2)
            model = timm.create_model('resnet18', pretrained=pretrained, num_classes=num_targets)
        elif use_model == 'res50':
            # model = torchvision.models.resnet50(pretrained=pretrained)
            # model.fc = nn.Linear(2048, 2)
            model = timm.create_model('resnet50', pretrained=pretrained, num_classes=num_targets)
        self.model = model
        
        # # Evaluation Losses
        self.loss_l1 = nn.L1Loss()
        self.loss_mse = nn.MSELoss()
        
        # this allows the trainer to show input and output sizes in the report (16 is just an example batch_size)
        self.example_input_array = torch.zeros(42, *data_shape)
                
        
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hparams.learning_rate)
        schedulers = []
        if self.hparams.scheduler == "one_cycle":
            steps_per_epoch = int(self.trainer.limit_train_batches * len(self.trainer.datamodule.train_dataloader())) # same as batches per epoch
            max_epochs = self.trainer.max_epochs
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.learning_rate,
                epochs=max_epochs,
                steps_per_epoch=steps_per_epoch,
                anneal_strategy="cos", # can be "linear" or "cos"(default)
            )
            # "interval: step" is required to let the scheduler update per step rather than epoch
            schedulers = [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
        elif self.hparams.scheduler == "cosine_annealing":
            steps_per_epoch = int(self.trainer.limit_train_batches * len(self.trainer.datamodule.train_dataloader())) # same as batches per epoch
            max_epochs = self.trainer.max_epochs
            max_steps = max_epochs * steps_per_epoch
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max_steps,
            )
            schedulers = [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
        return [optimizer], schedulers

    
    # # How to combine final activationa and final loss?
    # torch.nn.functional.cross_entropy(y_hat, y)
    #   - softmax (softmax ist schon innerhalb von nn.functional.cross_entropy implementiert, daher nicht extra softmax anwenden)
    # torch.nn.functional.one_hot(y, num_targets=self.hparams.num_targets).float()
    #   - sigmoid
    # torch.nn.functional.mse_loss(self.final_activation(y_hat), y)
    #   - relu
    #   - linear
    # torch.nn.functional.l1_loss(self.final_activation(y_hat), y)
    #   - relu
    #   - linear
    
    def final_activation(self, x):
        if self.hparams.activation == 'softmax':
            return x.softmax(1)
        elif self.hparams.activation == 'sigmoid':
            return x.sigmoid()
        elif self.hparams.activation == 'relu':
            return x.relu()
        elif self.hparams.activation == 'linear':
            return x
        else:
            return x
    
    def final_loss(self, y_hat, y):        
        # y_hat: prediction, y: target
        if self.hparams.loss_func == "ce": # softmax ist schon innerhalb von nn.functional.cross_entropy implementiert, daher nicht extra softmax anwenden
            return nn.functional.cross_entropy(y_hat, y)
        
        elif self.hparams.loss_func == "bce_logits": # sigmoid ist schon innerhalb von nn.functional.binary_cross_entropy implementiert, daher nicht extra sigmoid anwenden
            # y_onehot = torch.nn.functional.one_hot(y, num_targets=self.hparams.num_targets).float() * self.hparams.sigmoid_soft_factor
            return nn.functional.binary_cross_entropy_with_logits(y_hat, y)
        
        elif self.hparams.loss_func == "mse": # relu ist nicht innerhlab von mse_loss implementiert
            return nn.functional.mse_loss(self.final_activation(y_hat), y)
        
        elif self.hparams.loss_func == "l1": # relu ist nicht innerhlab von l1_loss implementiert
            return nn.functional.l1_loss(self.final_activation(y_hat), y)
        
        else:
            return nn.functional.mse_loss(self.final_activation(y_hat), y)
    
    def _forward(self, x):
        return self.model(x)
    
    
    def training_step(self, batch, batch_nb):
        img = batch['img']
        target = batch['target']
        
        pred = self._forward(img)
        loss = self.final_loss(pred, target)
        self.log("loss", loss)
        loss_l1 = self.loss_l1(self.final_activation(pred), target)
        self.log('loss_l1', loss_l1, on_step=False, on_epoch=True)
        loss_mse = self.loss_mse(self.final_activation(pred), target)
        self.log('loss_mse', loss_mse, on_step=False, on_epoch=True)

        return loss

    
    def validation_step(self, batch, batch_nb):
        img = batch['img']
        target = batch['target']
        
        pred = self._forward(img)
        loss = self.final_loss(pred, target)
        self.log("val_loss", loss)
        loss_l1 = self.loss_l1(self.final_activation(pred), target)
        self.log('val_loss_l1', loss_l1)
        loss_mse = self.loss_mse(self.final_activation(pred), target)
        self.log('val_loss_mse', loss_mse)

        return loss
    
    
    def test_step(self, batch, batch_nb):
        img = batch['img']
        target = batch['target']
        
        pred = self._forward(img)
        loss = self.final_loss(pred, target)
        self.log("test_loss", loss)
        loss_l1 = self.loss_l1(self.final_activation(pred), target)
        self.log('test_loss_l1', loss_l1)
        loss_mse = self.loss_mse(self.final_activation(pred), target)
        self.log('test_loss_mse', loss_mse)

        return loss


    def forward(self, x):
        # activation and target_transform is done here
        logits = self._forward(x)
        preds = self.final_activation(logits)
        return preds
