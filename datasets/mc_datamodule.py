#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 05 2023

@author: lukas
"""
import torch
import pytorch_lightning as pl
import numpy as np
from datasets.mc_dataset import MixCropDataset, MixCrop2ImagesDataset, MixCrop2Images2DatesDataset



class MixCropDataModule(pl.LightningDataModule):
    def __init__(self, img_dir, info_tab_path, wheat_target_path, bean_target_path, mix_target_path, wheat_treatment_path, bean_treatment_path, mix_treatment_path, data_name, data_time, batch_size, n_workers, transform_train, transform_test, target_type='total', target_transform=None, in_memory=False, val_test_shuffle=False):
        
        super().__init__()
        self.img_dir = img_dir
        self.info_tab_path = info_tab_path
        self.wheat_target_path = wheat_target_path
        self.bean_target_path = bean_target_path
        self.mix_target_path = mix_target_path
        self.wheat_treatment_path = wheat_treatment_path
        self.bean_treatment_path = bean_treatment_path
        self.mix_treatment_path = mix_treatment_path
        self.data_name = data_name
        self.data_time = data_time
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.transform_train = transform_train
        self.transform_test = transform_test
        self.target_type = target_type
        self.target_transform = target_transform
        self.in_memory = in_memory
        self.val_test_shuffle=val_test_shuffle
        
        self.params = {'batch_size': self.batch_size,
                       'n_workers': self.n_workers,
                       'img_dir': self.img_dir,
                       'transform_train': self.transform_train,
                       'transform_test': self.transform_test,
                       'val_test_shuffle': self.val_test_shuffle
                       }
    
    def prepare_data(self):
        pass
    
    def setup(self, stage=None):
        # 'or stage is None' to run both (fit and test) if its not specified
        if stage == "fit" or stage is None:
            self.train_data = MixCropDataset(self.img_dir+'train/', self.info_tab_path, self.wheat_target_path, self.bean_target_path, self.mix_target_path, self.wheat_treatment_path, self.bean_treatment_path, self.mix_treatment_path, self.data_name, self.data_time, transform=self.transform_train, target_type=self.target_type, target_transform=self.target_transform, in_memory=self.in_memory)
            self.val_data = MixCropDataset(self.img_dir+'val/', self.info_tab_path, self.wheat_target_path, self.bean_target_path, self.mix_target_path, self.wheat_treatment_path, self.bean_treatment_path, self.mix_treatment_path, self.data_name, self.data_time, transform=self.transform_test, target_type=self.target_type, target_transform=self.target_transform, in_memory=self.in_memory)
            self.data_dims = self.train_data[0]['img'].shape
        if stage == "test" or stage is None:
            self.test_data = MixCropDataset(self.img_dir+'test/', self.info_tab_path, self.wheat_target_path, self.bean_target_path, self.mix_target_path, self.wheat_treatment_path, self.bean_treatment_path, self.mix_treatment_path, self.data_name, self.data_time, transform=self.transform_test, target_type=self.target_type, target_transform=self.target_transform, in_memory=self.in_memory)
            self.data_dims = self.test_data[0]['img'].shape

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=True)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=self.val_test_shuffle)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=self.val_test_shuffle)
    
    
class MixCrop2ImagesDataModule(pl.LightningDataModule):
    def __init__(self, img_dir, info_tab_path, wheat_target_path, bean_target_path, mix_target_path, wheat_treatment_path, bean_treatment_path, mix_treatment_path, data_name, data_time, batch_size, n_workers, transform_train, transform_test, target_type='total', target_transform=None, in_memory=False, val_test_shuffle=False):
        super().__init__()
        self.img_dir = img_dir
        self.info_tab_path = info_tab_path
        self.wheat_target_path = wheat_target_path
        self.bean_target_path = bean_target_path
        self.mix_target_path = mix_target_path
        self.wheat_treatment_path = wheat_treatment_path
        self.bean_treatment_path = bean_treatment_path
        self.mix_treatment_path = mix_treatment_path
        self.data_name = data_name
        self.data_time = data_time
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.transform_train = transform_train
        self.transform_test = transform_test
        self.target_type = target_type
        self.target_transform = target_transform
        self.in_memory = in_memory
        self.val_test_shuffle=val_test_shuffle
        
        self.params = {'batch_size': self.batch_size,
                       'n_workers': self.n_workers,
                       'img_dir': self.img_dir,
                       'transform_train': self.transform_train,
                       'transform_test': self.transform_test,
                       'val_test_shuffle': self.val_test_shuffle
                       }
    
    def prepare_data(self):
        pass
    
    def setup(self, stage=None):
        # 'or stage is None' to run both (fit and test) if its not specified
        if stage == "fit" or stage is None:
            self.train_data = MixCrop2ImagesDataset(self.img_dir+'train/', self.info_tab_path, self.wheat_target_path, self.bean_target_path, self.mix_target_path, self.wheat_treatment_path, self.bean_treatment_path, self.mix_treatment_path, self.data_name, self.data_time, transform=self.transform_train, target_type=self.target_type, target_transform=self.target_transform, in_memory=self.in_memory)
            self.val_data = MixCrop2ImagesDataset(self.img_dir+'val/', self.info_tab_path, self.wheat_target_path, self.bean_target_path, self.mix_target_path, self.wheat_treatment_path, self.bean_treatment_path, self.mix_treatment_path, self.data_name, self.data_time, transform=self.transform_test, target_type=self.target_type, target_transform=self.target_transform, in_memory=self.in_memory)
            self.data_dims = self.train_data[0]['img_1'].shape
        if stage == "test" or stage is None:
            self.test_data = MixCrop2ImagesDataset(self.img_dir+'test/', self.info_tab_path, self.wheat_target_path, self.bean_target_path, self.mix_target_path, self.wheat_treatment_path, self.bean_treatment_path, self.mix_treatment_path, self.data_name, self.data_time, transform=self.transform_test, target_type=self.target_type, target_transform=self.target_transform, in_memory=self.in_memory)
            self.data_dims = self.test_data[0]['img_1'].shape

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=True)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=self.val_test_shuffle)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=self.val_test_shuffle)
    
    
class MixCrop2Images2DatesDataModule(pl.LightningDataModule):
    def __init__(self, img_dir, info_tab_path, wheat_target_path, bean_target_path, mix_target_path, wheat_treatment_path, bean_treatment_path, mix_treatment_path, data_name, data_time, date_in, date_out, batch_size, n_workers, transform_train, transform_test, target_type='total', target_transform=None, in_memory=False, val_test_shuffle=False): 
        super().__init__()
        self.img_dir = img_dir
        self.info_tab_path = info_tab_path
        self.wheat_target_path = wheat_target_path
        self.bean_target_path = bean_target_path
        self.mix_target_path = mix_target_path
        self.wheat_treatment_path = wheat_treatment_path
        self.bean_treatment_path = bean_treatment_path
        self.mix_treatment_path = mix_treatment_path
        self.data_name = data_name
        self.data_time = data_time
        self.date_in = date_in
        self.date_out = date_out
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.transform_train = transform_train
        self.transform_test = transform_test
        self.target_type = target_type
        self.target_transform = target_transform
        self.in_memory = in_memory
        self.val_test_shuffle=val_test_shuffle
        
        self.params = {'batch_size': self.batch_size,
                       'n_workers': self.n_workers,
                       'img_dir': self.img_dir,
                       'transform_train': self.transform_train,
                       'transform_test': self.transform_test,
                       'val_test_shuffle': self.val_test_shuffle
                       }
    
    def prepare_data(self):
        pass
    
    def setup(self, stage=None):
        # 'or stage is None' to run both (fit and test) if its not specified
        if stage == "fit" or stage is None:
            self.train_data = MixCrop2Images2DatesDataset(self.img_dir+'train/', self.info_tab_path, self.wheat_target_path, self.bean_target_path, self.mix_target_path, self.wheat_treatment_path, self.bean_treatment_path, self.mix_treatment_path, self.data_name, self.data_time, self.date_in, self.date_out, transform=self.transform_train, target_type=self.target_type, target_transform=self.target_transform, in_memory=self.in_memory)
            self.val_data = MixCrop2Images2DatesDataset(self.img_dir+'val/', self.info_tab_path, self.wheat_target_path, self.bean_target_path, self.mix_target_path, self.wheat_treatment_path, self.bean_treatment_path, self.mix_treatment_path, self.data_name, self.data_time, self.date_in, self.date_out, transform=self.transform_test, target_type=self.target_type, target_transform=self.target_transform, in_memory=self.in_memory)
            self.data_dims = self.train_data[0]['img_1'].shape
        if stage == "test" or stage is None:
            self.test_data = MixCrop2Images2DatesDataset(self.img_dir+'test/', self.info_tab_path, self.wheat_target_path, self.bean_target_path, self.mix_target_path, self.wheat_treatment_path, self.bean_treatment_path, self.mix_treatment_path, self.data_name, self.data_time, self.date_in, self.date_out, transform=self.transform_test, target_type=self.target_type, target_transform=self.target_transform, in_memory=self.in_memory)
            self.data_dims = self.test_data[0]['img_1'].shape

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=True)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=self.val_test_shuffle)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=self.val_test_shuffle)
