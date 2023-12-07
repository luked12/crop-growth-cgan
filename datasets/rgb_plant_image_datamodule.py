import torch
import pytorch_lightning as pl
import numpy as np
from datasets.rgb_plant_image_dataset import RGBPlantImageDataset, RGBPlant2ImagesDataset, RGBPlant2Images2DatesDataset



class RGBPlantImageDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, n_workers, img_dir, data_name, data_time, transform_train, transform_test, in_memory=True, val_test_shuffle=False):
        super().__init__()
        self.batch_size=batch_size
        self.n_workers=n_workers
        self.img_dir=img_dir
        self.data_name = data_name
        self.data_time = data_time
        self.transform_train=transform_train
        self.transform_test=transform_test
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
            self.train_data = RGBPlantImageDataset(self.img_dir+'train/', self.data_name, self.data_time, self.transform_train, in_memory=self.in_memory)
            self.val_data = RGBPlantImageDataset(self.img_dir+'val/', self.data_name, self.data_time, self.transform_test, in_memory=self.in_memory)
            self.data_dims = self.train_data[0]['img'].shape
        if stage == "test" or stage is None:
            self.test_data = RGBPlantImageDataset(self.img_dir+'test/', self.data_name, self.data_time, self.transform_test, in_memory=self.in_memory)
            self.data_dims = self.test_data[0]['img'].shape

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=True)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=self.val_test_shuffle)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=self.val_test_shuffle)



class RGBPlant2ImagesDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, n_workers, img_dir, data_name, data_time, transform_train, transform_test, in_memory=True, val_test_shuffle=False):
        super().__init__()
        self.batch_size=batch_size
        self.n_workers=n_workers
        self.img_dir=img_dir
        self.data_name = data_name
        self.data_time = data_time
        self.transform_train=transform_train
        self.transform_test=transform_test
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
            self.train_data = RGBPlant2ImagesDataset(self.img_dir+'train/', self.data_name, self.data_time, self.transform_train, in_memory=self.in_memory)
            self.val_data = RGBPlant2ImagesDataset(self.img_dir+'val/', self.data_name, self.data_time, self.transform_test, in_memory=self.in_memory)
            self.data_dims = self.train_data[0]['img_1'].shape
        if stage == "test" or stage is None:
            self.test_data = RGBPlant2ImagesDataset(self.img_dir+'test/', self.data_name, self.data_time, self.transform_test, in_memory=self.in_memory)
            self.data_dims = self.test_data[0]['img_1'].shape

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=True)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=self.val_test_shuffle)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=self.val_test_shuffle)
    


class RGBPlant2Images2DatesDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, n_workers, img_dir, data_name, data_time, transform_train, transform_test, in_memory=True, val_test_shuffle=False):
        super().__init__()
        self.batch_size=batch_size
        self.n_workers=n_workers
        self.img_dir=img_dir
        self.data_name = data_name
        self.data_time = data_time
        self.transform_train=transform_train
        self.transform_test=transform_test
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
            self.train_data = RGBPlant2Images2DatesDataset(self.img_dir+'train/', self.data_name, self.data_time, self.transform_train, in_memory=self.in_memory)
            self.val_data = RGBPlant2Images2DatesDataset(self.img_dir+'val/', self.data_name, self.data_time, self.transform_test, in_memory=self.in_memory)
            self.data_dims = self.train_data[0]['img_1'].shape
        if stage == "test" or stage is None:
            self.test_data = RGBPlant2Images2DatesDataset(self.img_dir+'test/', self.data_name, self.data_time, self.transform_test, in_memory=self.in_memory)
            self.data_dims = self.test_data[0]['img_1'].shape

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=True)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=self.val_test_shuffle)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=self.val_test_shuffle)