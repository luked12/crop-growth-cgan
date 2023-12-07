#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 12:37:43 2021

@author: lukas
"""

from __future__ import print_function, division
import os
import torch
import pandas as pd
import torchvision
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import random
import datetime
import re
from utils import utils



# # Define global variables: mapping of treatment to cls idx
map_treat_cls_3 = {'Mix': 0, 'SW': 1,'FB': 2,}
map_treat_cls_76 = {'FB_A_H': 0,
                    'FB_A_L': 1,
                    'FB_B_H': 2,
                    'FB_B_L': 3,
                    'SW_1_H': 4,
                    'SW_1_L': 5,
                    'SW_2_H': 6,
                    'SW_2_L': 7,
                    'SW_3_H': 8,
                    'SW_3_L': 9,
                    'SW_4_H': 10,
                    'SW_4_L': 11,
                    'SW_5_H': 12,
                    'SW_5_L': 13,
                    'SW_6_H': 14,
                    'SW_6_L': 15,
                    'SW_7_H': 16,
                    'SW_7_L': 17,
                    'SW_8_H': 18,
                    'SW_8_L': 19,
                    'SW_9_H': 20,
                    'SW_9_L': 21,
                    'SW_10_H': 22,
                    'SW_10_L': 23,
                    'SW_11_H': 24,
                    'SW_11_L': 25,
                    'SW_12_H': 26,
                    'SW_12_L': 27,
                    'Mix_A_1_H': 28,
                    'Mix_A_1_L': 29,
                    'Mix_A_2_H': 30,
                    'Mix_A_2_L': 31,
                    'Mix_A_3_H': 32,
                    'Mix_A_3_L': 33,
                    'Mix_A_4_H': 34,
                    'Mix_A_4_L': 35,
                    'Mix_A_5_H': 36,
                    'Mix_A_5_L': 37,
                    'Mix_A_6_H': 38,
                    'Mix_A_6_L': 39,
                    'Mix_A_7_H': 40,
                    'Mix_A_7_L': 41,
                    'Mix_A_8_H': 42,
                    'Mix_A_8_L': 43,
                    'Mix_A_9_H': 44,
                    'Mix_A_9_L': 45,
                    'Mix_A_10_H': 46,
                    'Mix_A_10_L': 47,
                    'Mix_A_11_H': 48,
                    'Mix_A_11_L': 49,
                    'Mix_A_12_H': 50,
                    'Mix_A_12_L': 51,
                    'Mix_B_1_H': 52,
                    'Mix_B_1_L': 53,
                    'Mix_B_2_H': 54,
                    'Mix_B_2_L': 55,
                    'Mix_B_3_H': 56,
                    'Mix_B_3_L': 57,
                    'Mix_B_4_H': 58,
                    'Mix_B_4_L': 59,
                    'Mix_B_5_H': 60,
                    'Mix_B_5_L': 61,
                    'Mix_B_6_H': 62,
                    'Mix_B_6_L': 63,
                    'Mix_B_7_H': 64,
                    'Mix_B_7_L': 65,
                    'Mix_B_8_H': 66,
                    'Mix_B_8_L': 67,
                    'Mix_B_9_H': 68,
                    'Mix_B_9_L': 69,
                    'Mix_B_10_H': 70,
                    'Mix_B_10_L': 71,
                    'Mix_B_11_H': 72,
                    'Mix_B_11_L': 73,
                    'Mix_B_12_H': 74,
                    'Mix_B_12_L': 75}


def get_time_from_path(img_path, data_name):
    '''
    Function to get a time object from an img_path of a dataset
    You need the data_name, since in each dataset the time is indicated differently
    
    Args: 
        img_path : str
            Path of img in dataset
        data_name :  str
            name of dataset
    
    Returns: 
        time_obj :  datetime.datetime
            time object of image
    '''
    
    if data_name == 'abd' or data_name == 'abdc':
        time_string = img_path[-32:-12]
        time_obj = datetime.datetime.strptime(time_string, '%Y-%m-%d--%H-%M-%S')
    elif data_name == 'grf':
        split_pos = [m.start() for m in re.finditer('/', img_path)]
        time_string = img_path[split_pos[-1]+1:split_pos[-1]+11]
        time_obj = datetime.datetime.strptime(time_string, '%Y_%m_%d')
    elif data_name == 'mix':
        split_pos = [m.start() for m in re.finditer('/', img_path)]
        time_string = img_path[split_pos[-1]+9:split_pos[-1]+19]
        time_obj = datetime.datetime.strptime(time_string, '%Y_%m_%d')
    elif data_name == 'mix-wg':
        split_pos = [m.start() for m in re.finditer('/', img_path)]
        time_string = img_path[split_pos[-1]+8:split_pos[-1]+18]
        time_obj = datetime.datetime.strptime(time_string, '%Y_%m_%d')
    elif data_name == 'dnt':
        time_string = img_path[-19:-9]
        time_obj = datetime.datetime.strptime(time_string, '%Y_%m_%d')
    else:
        print('Error: Wrong dataset_name:',data_name )
    
    return time_obj


def get_ratio_from_total_target(target):
    ratio_B = target[0]/(target[0]+target[1])
    ratio_W = target[1]/(target[0]+target[1])
    target = torch.tensor((ratio_B,ratio_W)).float()
    return target
    

def get_plotNr_from_path(img_path, data_name):
    '''
    Function to get the PlotNumber out of the path

    Args: 
        img_path : str
            Path of img in dataset

    Returns: 
        plotNr : str
            plotNr
    '''
    split_pos = [m.start() for m in re.finditer('/', img_path)]
    if data_name == 'mix':    
        plotNr = img_path[split_pos[-1]+5:split_pos[-1]+8]
    elif data_name == 'mix-wg':
        plotNr = img_path[split_pos[-1]+4:split_pos[-1]+7]

    return plotNr


def pil_loader(path):
    '''
    ---------------------------------------------------------------------------
    code reference: https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
    '''
    
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            if len(img.getbands())==1:
                return img.convert('L')
            else:
                return img.convert('RGB')
            
            
def accimage_loader(path):
    '''
    ---------------------------------------------------------------------------
    code reference: https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
    '''
    
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

    
def default_image_loader():
    '''
    ---------------------------------------------------------------------------
    code reference: https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
    '''
    
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader
            
            
def find_seqs(seq_root):
    '''
    This function finds subdirectories containing the full length seqs in the root seq folder. 
    
    Args: 
        seq_root : str
            Path to root directory of seq folders.
            
    Returns: 
        seq_names : list
            List of seq names.
        seq_idx : dict
            Dict with items (seq_names, seq_idx)
        
    ---------------------------------------------------------------------------
    code adapted from: https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py        
    '''
    
    seq_names = [d for d in os.listdir(seq_root) if os.path.isdir(os.path.join(seq_root, d))]
    seq_names.sort()
    seq_idx = {seq_names[i]: i for i in range(len(seq_names))}
    return seq_names, seq_idx      


class MixCropDataset(Dataset):
    def __init__(self, img_dir, info_tab_path, wheat_target_path, bean_target_path, mix_target_path, wheat_treatment_path, bean_treatment_path, mix_treatment_path, data_name, data_time, transform, target_type='total', target_transform=None, in_memory=False):
        self.transform = transform
        self.in_memory = in_memory
        self.image_loader = default_image_loader()

        # # get factor to unit
        factor_to_unit = utils.get_seconds_factor_to_time_unit(data_time['time_unit'])

        # # initialize lists
        self.all_paths = []
        self.all_names = []
        self.all_plotNr = []
        self.all_id = []
        self.all_times = []
        self.all_targets = []
        self.all_orig_trt = []
        self.all_orig_dense = []
        self.all_actualMix = []
        self.all_label = []
        if self.in_memory:
            self.data = []
        
        # # get paths, names and idx
        seq_names, seq_idx = find_seqs(img_dir)
        for name in seq_names:
            plant_paths = utils.getListOfImgFiles(img_dir+name)
            plant_names = [name] * len(plant_paths)
            plant_plotNr = [int(get_plotNr_from_path(plant_paths[0], data_name))] * len(plant_paths)
            plant_id = [seq_idx[name]] * len(plant_paths)
            self.all_paths += plant_paths
            self.all_names += plant_names
            self.all_plotNr += plant_plotNr
            self.all_id += plant_id
           
        # # get info table and simulation tables  
        info_tab_df = pd.read_csv(info_tab_path)
        wheat_target_df = pd.read_csv(wheat_target_path)
        bean_target_df = pd.read_csv(bean_target_path)
        mix_target_df = pd.read_csv(mix_target_path)
        wheat_treatment_df = pd.read_csv(wheat_treatment_path)
        bean_treatment_df = pd.read_csv(bean_treatment_path)
        mix_treatment_df = pd.read_csv(mix_treatment_path)
           
        # # map treatment to class labels (coarse [3 classes] or fine [76 classes])
        # info_tab_df['labels'] = info_tab_df['actualMix']
        # info_tab_df['labels'] = info_tab_df['labels'].map(map_treat_cls_3)
        info_tab_df['labels'] = info_tab_df['orig_trt']
        info_tab_df['labels'] = info_tab_df['labels'].map(map_treat_cls_76)
        
        # # change dates to DAP/DAS in column "CURRENT.DATE" 
        for index, value in enumerate(wheat_target_df['CURRENT.DATE']):
            time = datetime.datetime.strptime(value, '%d.%m.%Y')
            timedelta_ToStart = time-data_time['time_start'] # timedelta object
            timedelta = round(timedelta_ToStart.total_seconds()/factor_to_unit) # int
            wheat_target_df.loc[index, 'CURRENT.DATE'] = timedelta
        for index, value in enumerate(bean_target_df['CURRENT.DATE']):
            time = datetime.datetime.strptime(value, '%d.%m.%Y')
            timedelta_ToStart = time-data_time['time_start'] # timedelta object
            timedelta = round(timedelta_ToStart.total_seconds()/factor_to_unit) # int
            bean_target_df.loc[index, 'CURRENT.DATE'] = timedelta
        for index, value in enumerate(mix_target_df['CURRENT.DATE']):
            time = datetime.datetime.strptime(value, '%d.%m.%Y')
            timedelta_ToStart = time-data_time['time_start'] # timedelta object
            timedelta = round(timedelta_ToStart.total_seconds()/factor_to_unit) # int
            mix_target_df.loc[index, 'CURRENT.DATE'] = timedelta
           
        # # split transform as the first three (ToTensor, Normalize, and Resize) can be applied in init (if self.in_memory)
        if self.in_memory:
            init_transform = []
            getItem_transform = []
            for idx, transforms in enumerate(self.transform.transforms):
                if idx<=2:
                    init_transform.append(transforms)
                else:    
                    getItem_transform.append(transforms)
            init_transform = torchvision.transforms.Compose(init_transform)
            self.transform = torchvision.transforms.Compose(getItem_transform)
           
        # # iterate over all paths and load times, targets and - if self.in_memory - img data    
        for (plotNr, path) in zip(self.all_plotNr,self.all_paths):
            # # times
            time = get_time_from_path(path, data_name)
            timedelta_ToStart = time-data_time['time_start'] # timedelta object
            timedelta = round(timedelta_ToStart.total_seconds()/factor_to_unit) # int
            self.all_times+=[timedelta]
            
            # # get correct row of data frmae
            info_row_df = info_tab_df.loc[info_tab_df['PlotNo'] == plotNr]
            # # general infos regarding the plot
            orig_trt = info_row_df['orig_trt'].values[0]
            self.all_orig_trt.append(orig_trt)
            self.all_orig_dense.append(info_row_df['orig_dens'].values[0])
            actualMix = info_row_df['actualMix'].values[0]
            self.all_actualMix.append(actualMix)
            self.all_label.append(int(info_row_df['labels'].values[0]))
            
            # # Extract correct target from process-based simulation files
            site = 'CKA' if self.all_names[0][0] == 'C' else 'WG' 
            if actualMix == 'Mix':
                simulation_id = mix_treatment_df.loc[mix_treatment_df['orig_trt'] == orig_trt].loc[mix_treatment_df['Location'] == site]['projectid'].values[0]
                biomass_W = mix_target_df.loc[mix_target_df['projectid'] == simulation_id].loc[mix_target_df['CURRENT.DATE'] == timedelta]['AGBG_1_t_ha'].values[0]
                biomass_B = mix_target_df.loc[mix_target_df['projectid'] == simulation_id].loc[mix_target_df['CURRENT.DATE'] == timedelta]['AGBG_2_t_ha'].values[0]
            elif actualMix == 'FB':
                simulation_id = bean_treatment_df.loc[bean_treatment_df['orig_trt'] == orig_trt].loc[bean_treatment_df['Location'] == site]['projectid'].values[0]
                biomass_W = bean_target_df.loc[bean_target_df['projectid'] == simulation_id].loc[bean_target_df['CURRENT.DATE'] == timedelta]['AGBG_1_t_ha'].values[0]
                biomass_B = bean_target_df.loc[bean_target_df['projectid'] == simulation_id].loc[bean_target_df['CURRENT.DATE'] == timedelta]['AGBG_2_t_ha'].values[0]
            elif actualMix == 'SW':
                simulation_id = wheat_treatment_df.loc[wheat_treatment_df['orig_trt'] == orig_trt].loc[wheat_treatment_df['Location'] == site]['projectid'].values[0]
                biomass_W = wheat_target_df.loc[wheat_target_df['projectid'] == simulation_id].loc[wheat_target_df['CURRENT.DATE'] == timedelta]['AGBG_1_t_ha'].values[0]
                biomass_B = wheat_target_df.loc[wheat_target_df['projectid'] == simulation_id].loc[wheat_target_df['CURRENT.DATE'] == timedelta]['AGBG_2_t_ha'].values[0]
            target = torch.tensor((biomass_B,biomass_W)).float()
            
            # # transform target
            if target_type == 'ratio':
                target = get_ratio_from_total_target(target)
            if target_transform is not None:
                target = torch.div(target, target_transform)
            self.all_targets.append(target)
            
            # # img data
            if self.in_memory:
                self.data.append(init_transform(self.image_loader(path)))
            
    def __len__(self):
        return len(self.all_paths)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # # access or load images
        if self.in_memory:
            image = self.data[idx]
        else:
            image = self.image_loader(self.all_paths[idx])
        
        # # transform img
        if self.transform:
            image = self.transform(image)
        
        # # build sample dict
        sample = {'id': self.all_id[idx],
                  'img': image,
                  'name': self.all_names[idx],   
                  'time': self.all_times[idx], 
                  'biomass': self.all_targets[idx],
                  'plotNr': self.all_plotNr[idx],
                  'orig_trt': self.all_orig_trt[idx],
                  'orig_dense': self.all_orig_dense[idx],
                  'actualMix': self.all_actualMix[idx],
                  'label': self.all_label[idx],
                  }
        
        return sample



class MixCrop2ImagesDataset(Dataset):
    def __init__(self, img_dir, info_tab_path, wheat_target_path, bean_target_path, mix_target_path, wheat_treatment_path, bean_treatment_path, mix_treatment_path, data_name, data_time, transform, target_type='total', target_transform=None, in_memory=False):
        self.transform = transform
        self.in_memory = in_memory
        self.image_loader = default_image_loader()

        # # get factor to unit
        factor_to_unit = utils.get_seconds_factor_to_time_unit(data_time['time_unit'])

        # # initialize lists
        self.all_paths = []
        self.all_names = []
        self.all_plotNr = []
        self.all_id = []
        self.all_times = []
        self.all_targets = []
        self.all_orig_trt = []
        self.all_orig_dense = []
        self.all_actualMix = []
        self.all_label = []
        if self.in_memory:
            self.data = []
        
        # # get paths, names and idx
        seq_names, seq_idx = find_seqs(img_dir)
        for name in seq_names:
            plant_paths = utils.getListOfImgFiles(img_dir+name)
            plant_names = [name] * len(plant_paths)
            plant_plotNr = [int(get_plotNr_from_path(plant_paths[0], data_name))] * len(plant_paths)
            plant_id = [seq_idx[name]] * len(plant_paths)
            self.all_paths += plant_paths
            self.all_names += plant_names
            self.all_plotNr += plant_plotNr
            self.all_id += plant_id
           
        # # get info table and simulation tables  
        info_tab_df = pd.read_csv(info_tab_path)
        wheat_target_df = pd.read_csv(wheat_target_path)
        bean_target_df = pd.read_csv(bean_target_path)
        mix_target_df = pd.read_csv(mix_target_path)
        wheat_treatment_df = pd.read_csv(wheat_treatment_path)
        bean_treatment_df = pd.read_csv(bean_treatment_path)
        mix_treatment_df = pd.read_csv(mix_treatment_path)
           
        # # map treatment to class labels (coarse [3 classes] or fine [76 classes])
        # info_tab_df['labels'] = info_tab_df['actualMix']
        # info_tab_df['labels'] = info_tab_df['labels'].map(map_treat_cls_3)
        info_tab_df['labels'] = info_tab_df['orig_trt']
        info_tab_df['labels'] = info_tab_df['labels'].map(map_treat_cls_76)
        
        # # change dates to DAP/DAS in column "CURRENT.DATE" 
        for index, value in enumerate(wheat_target_df['CURRENT.DATE']):
            time = datetime.datetime.strptime(value, '%d.%m.%Y')
            timedelta_ToStart = time-data_time['time_start'] # timedelta object
            timedelta = round(timedelta_ToStart.total_seconds()/factor_to_unit) # int
            wheat_target_df.loc[index, 'CURRENT.DATE'] = timedelta
        for index, value in enumerate(bean_target_df['CURRENT.DATE']):
            time = datetime.datetime.strptime(value, '%d.%m.%Y')
            timedelta_ToStart = time-data_time['time_start'] # timedelta object
            timedelta = round(timedelta_ToStart.total_seconds()/factor_to_unit) # int
            bean_target_df.loc[index, 'CURRENT.DATE'] = timedelta
        for index, value in enumerate(mix_target_df['CURRENT.DATE']):
            time = datetime.datetime.strptime(value, '%d.%m.%Y')
            timedelta_ToStart = time-data_time['time_start'] # timedelta object
            timedelta = round(timedelta_ToStart.total_seconds()/factor_to_unit) # int
            mix_target_df.loc[index, 'CURRENT.DATE'] = timedelta
           
        # # split transform as the first three (ToTensor, Normalize, and Resize) can be applied in init (if self.in_memory)
        if self.in_memory:
            init_transform = []
            getItem_transform = []
            for idx, transforms in enumerate(self.transform.transforms):
                if idx<=2:
                    init_transform.append(transforms)
                else:    
                    getItem_transform.append(transforms)
            init_transform = torchvision.transforms.Compose(init_transform)
            self.transform = torchvision.transforms.Compose(getItem_transform)
           
        # # iterate over all paths and load times, targets and - if self.in_memory - img data    
        for (plotNr, path) in zip(self.all_plotNr,self.all_paths):
            # # times
            time = get_time_from_path(path, data_name)
            timedelta_ToStart = time-data_time['time_start'] # timedelta object
            timedelta = round(timedelta_ToStart.total_seconds()/factor_to_unit) # int
            self.all_times+=[timedelta]
            
            # # get correct row of data frmae
            info_row_df = info_tab_df.loc[info_tab_df['PlotNo'] == plotNr]
            # # general infos regarding the plot
            orig_trt = info_row_df['orig_trt'].values[0]
            self.all_orig_trt.append(orig_trt)
            self.all_orig_dense.append(info_row_df['orig_dens'].values[0])
            actualMix = info_row_df['actualMix'].values[0]
            self.all_actualMix.append(actualMix)
            self.all_label.append(int(info_row_df['labels'].values[0]))
            
            # # Extract correct target from process-based simulation files
            site = 'CKA' if self.all_names[0][0] == 'C' else 'WG' 
            if actualMix == 'Mix':
                simulation_id = mix_treatment_df.loc[mix_treatment_df['orig_trt'] == orig_trt].loc[mix_treatment_df['Location'] == site]['projectid'].values[0]
                biomass_W = mix_target_df.loc[mix_target_df['projectid'] == simulation_id].loc[mix_target_df['CURRENT.DATE'] == timedelta]['AGBG_1_t_ha'].values[0]
                biomass_B = mix_target_df.loc[mix_target_df['projectid'] == simulation_id].loc[mix_target_df['CURRENT.DATE'] == timedelta]['AGBG_2_t_ha'].values[0]
            elif actualMix == 'FB':
                simulation_id = bean_treatment_df.loc[bean_treatment_df['orig_trt'] == orig_trt].loc[bean_treatment_df['Location'] == site]['projectid'].values[0]
                biomass_W = bean_target_df.loc[bean_target_df['projectid'] == simulation_id].loc[bean_target_df['CURRENT.DATE'] == timedelta]['AGBG_1_t_ha'].values[0]
                biomass_B = bean_target_df.loc[bean_target_df['projectid'] == simulation_id].loc[bean_target_df['CURRENT.DATE'] == timedelta]['AGBG_2_t_ha'].values[0]
            elif actualMix == 'SW':
                simulation_id = wheat_treatment_df.loc[wheat_treatment_df['orig_trt'] == orig_trt].loc[wheat_treatment_df['Location'] == site]['projectid'].values[0]
                biomass_W = wheat_target_df.loc[wheat_target_df['projectid'] == simulation_id].loc[wheat_target_df['CURRENT.DATE'] == timedelta]['AGBG_1_t_ha'].values[0]
                biomass_B = wheat_target_df.loc[wheat_target_df['projectid'] == simulation_id].loc[wheat_target_df['CURRENT.DATE'] == timedelta]['AGBG_2_t_ha'].values[0]
            target = torch.tensor((biomass_B,biomass_W)).float()
            
            # # transform target
            if target_type == 'ratio':
                target = get_ratio_from_total_target(target)
            if target_transform is not None:
                target = torch.div(target, target_transform)
            self.all_targets.append(target)
            
            # # img data
            if self.in_memory:
                self.data.append(init_transform(self.image_loader(path)))
                
        # # assign plant to sequence        
        self.plant_seq_assignment = []
        for plant_id in self.all_id:
            other_idx = np.where(np.array(self.all_id)==plant_id)[0]
            self.plant_seq_assignment.append(other_idx)
            
    def __len__(self):
        return len(self.all_paths)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # # get random other idx of the same plant
        rand_idx = random.randint(0,len(self.plant_seq_assignment[idx])-1)
        idx2 = self.plant_seq_assignment[idx][rand_idx]
        
        # # access or load images
        if self.in_memory:
            image_1 = self.data[idx]
            image_2 = self.data[idx2]
        else:
            image_1 = self.image_loader(self.all_paths[idx])
            image_2 = self.image_loader(self.all_paths[idx2])
        
        # # transform both imgs (attention: set seed to transform both imgs in the same way)
        seed = np.random.randint(245985462)
        if self.transform:
            torch.manual_seed(seed)
            random.seed(seed)
            image_1 = self.transform(image_1)
            torch.manual_seed(seed)
            random.seed(seed)
            image_2 = self.transform(image_2)
        
        # # build sample dict
        sample = {'id': self.all_id[idx],
                  'img_1': image_1,
                  'name_1': self.all_names[idx],   
                  'time_1': self.all_times[idx], 
                  'biomass_1': self.all_targets[idx],
                  'img_2': image_2,
                  'name_2': self.all_names[idx2],   
                  'time_2': self.all_times[idx2],
                  'biomass_2': self.all_targets[idx2],
                  'plotNr': self.all_plotNr[idx],
                  'orig_trt': self.all_orig_trt[idx],
                  'orig_dense': self.all_orig_dense[idx],
                  'actualMix': self.all_actualMix[idx],
                  'label': self.all_label[idx],
                  }
        
        return sample
    
    
    
class MixCrop2Images2DatesDataset(Dataset):
    def __init__(self, img_dir, info_tab_path, wheat_target_path, bean_target_path, mix_target_path, wheat_treatment_path, bean_treatment_path, mix_treatment_path, data_name, data_time, date_in, date_out, transform, target_type='total', target_transform=None, in_memory=False):
        self.transform = transform
        self.in_memory = in_memory
        self.image_loader = default_image_loader()

        # # get factor to unit
        factor_to_unit = utils.get_seconds_factor_to_time_unit(data_time['time_unit'])

        # # initialize lists
        self.all_in_paths = []
        self.all_out_paths = []
        self.all_names = []
        self.all_plotNr = []
        self.all_id = []
        self.all_targets_in = []
        self.all_targets_out = []
        self.all_orig_trt = []
        self.all_orig_dense = []
        self.all_actualMix = []
        self.all_label = []
        if self.in_memory:
            self.data_in = []
            self.data_out = []
            
        # # paths, names and idx
        seq_names, seq_idx = find_seqs(img_dir)
        for name in seq_names:
            plant_paths = utils.getListOfImgFiles(img_dir+name)
            plant_name = name
            plant_plotNr = int(get_plotNr_from_path(plant_paths[0], data_name))
            plant_id = seq_idx[name]
            filter_in_idx = [idx for idx, item in enumerate(plant_paths) if date_in in item]
            filter_out_idx = [idx for idx, item in enumerate(plant_paths) if date_out in item]
            if filter_in_idx and filter_out_idx:
                self.all_in_paths.append(plant_paths[filter_in_idx[0]])
                self.all_out_paths.append(plant_paths[filter_out_idx[0]])
                self.all_names.append(plant_name)
                self.all_plotNr.append(plant_plotNr)
                self.all_id.append(plant_id)
        
        # # get info table and simulation tables  
        info_tab_df = pd.read_csv(info_tab_path)
        wheat_target_df = pd.read_csv(wheat_target_path)
        bean_target_df = pd.read_csv(bean_target_path)
        mix_target_df = pd.read_csv(mix_target_path)
        wheat_treatment_df = pd.read_csv(wheat_treatment_path)
        bean_treatment_df = pd.read_csv(bean_treatment_path)
        mix_treatment_df = pd.read_csv(mix_treatment_path)
           
        # # map treatment to class labels (coarse [3 classes] or fine [76 classes])
        # info_tab_df['labels'] = info_tab_df['actualMix']
        # info_tab_df['labels'] = info_tab_df['labels'].map(map_treat_cls_3)
        info_tab_df['labels'] = info_tab_df['orig_trt']
        info_tab_df['labels'] = info_tab_df['labels'].map(map_treat_cls_76)
        
        # # change dates to DAP/DAS in column "CURRENT.DATE" 
        for index, value in enumerate(wheat_target_df['CURRENT.DATE']):
            time = datetime.datetime.strptime(value, '%d.%m.%Y')
            timedelta_ToStart = time-data_time['time_start'] # timedelta object
            timedelta = round(timedelta_ToStart.total_seconds()/factor_to_unit) # int
            wheat_target_df.loc[index, 'CURRENT.DATE'] = timedelta
        for index, value in enumerate(bean_target_df['CURRENT.DATE']):
            time = datetime.datetime.strptime(value, '%d.%m.%Y')
            timedelta_ToStart = time-data_time['time_start'] # timedelta object
            timedelta = round(timedelta_ToStart.total_seconds()/factor_to_unit) # int
            bean_target_df.loc[index, 'CURRENT.DATE'] = timedelta
        for index, value in enumerate(mix_target_df['CURRENT.DATE']):
            time = datetime.datetime.strptime(value, '%d.%m.%Y')
            timedelta_ToStart = time-data_time['time_start'] # timedelta object
            timedelta = round(timedelta_ToStart.total_seconds()/factor_to_unit) # int
            mix_target_df.loc[index, 'CURRENT.DATE'] = timedelta
           
        # # split transform as the first three (ToTensor, Normalize, and Resize) can be applied in init (if self.in_memory)
        if self.in_memory:
            init_transform = []
            getItem_transform = []
            for idx, transforms in enumerate(self.transform.transforms):
                if idx<=2:
                    init_transform.append(transforms)
                else:    
                    getItem_transform.append(transforms)
            init_transform = torchvision.transforms.Compose(init_transform)
            self.transform = torchvision.transforms.Compose(getItem_transform)
            
        # # load time_in and _out
        time = datetime.datetime.strptime(date_in, '%Y_%m_%d')
        timedelta_ToStart = time-data_time['time_start'] # timedelta object
        self.time_in = round(timedelta_ToStart.total_seconds()/factor_to_unit) # int
        time = datetime.datetime.strptime(date_out, '%Y_%m_%d')
        timedelta_ToStart = time-data_time['time_start'] # timedelta object
        self.time_out = round(timedelta_ToStart.total_seconds()/factor_to_unit) # int
            
        
        # # iterate over all paths and load times, targets and - if self.in_memory - img data    
        for (plotNr, in_path, out_path) in zip(self.all_plotNr,self.all_in_paths,self.all_out_paths):          
            # # get correct row of data frmae
            info_row_df = info_tab_df.loc[info_tab_df['PlotNo'] == plotNr]
            # # general infos regarding the plot
            orig_trt = info_row_df['orig_trt'].values[0]
            self.all_orig_trt.append(orig_trt)
            self.all_orig_dense.append(info_row_df['orig_dens'].values[0])
            actualMix = info_row_df['actualMix'].values[0]
            self.all_actualMix.append(actualMix)
            self.all_label.append(int(info_row_df['labels'].values[0]))
            
            # # Extract correct target from process-based simulation files
            site = 'CKA' if self.all_names[0][0] == 'C' else 'WG' 
            if actualMix == 'Mix':
                simulation_id = mix_treatment_df.loc[mix_treatment_df['orig_trt'] == orig_trt].loc[mix_treatment_df['Location'] == site]['projectid'].values[0]
                biomass_W_in = mix_target_df.loc[mix_target_df['projectid'] == simulation_id].loc[mix_target_df['CURRENT.DATE'] == self.time_in]['AGBG_1_t_ha'].values[0]
                biomass_B_in = mix_target_df.loc[mix_target_df['projectid'] == simulation_id].loc[mix_target_df['CURRENT.DATE'] == self.time_in]['AGBG_2_t_ha'].values[0]
                biomass_W_out = mix_target_df.loc[mix_target_df['projectid'] == simulation_id].loc[mix_target_df['CURRENT.DATE'] == self.time_out]['AGBG_1_t_ha'].values[0]
                biomass_B_out = mix_target_df.loc[mix_target_df['projectid'] == simulation_id].loc[mix_target_df['CURRENT.DATE'] == self.time_out]['AGBG_2_t_ha'].values[0]
            elif actualMix == 'FB':
                simulation_id = bean_treatment_df.loc[bean_treatment_df['orig_trt'] == orig_trt].loc[bean_treatment_df['Location'] == site]['projectid'].values[0]
                biomass_W_in = bean_target_df.loc[bean_target_df['projectid'] == simulation_id].loc[bean_target_df['CURRENT.DATE'] == self.time_in]['AGBG_1_t_ha'].values[0]
                biomass_B_in = bean_target_df.loc[bean_target_df['projectid'] == simulation_id].loc[bean_target_df['CURRENT.DATE'] == self.time_in]['AGBG_2_t_ha'].values[0]
                biomass_W_out = bean_target_df.loc[bean_target_df['projectid'] == simulation_id].loc[bean_target_df['CURRENT.DATE'] == self.time_out]['AGBG_1_t_ha'].values[0]
                biomass_B_out = bean_target_df.loc[bean_target_df['projectid'] == simulation_id].loc[bean_target_df['CURRENT.DATE'] == self.time_out]['AGBG_2_t_ha'].values[0]
            elif actualMix == 'SW':
                simulation_id = wheat_treatment_df.loc[wheat_treatment_df['orig_trt'] == orig_trt].loc[wheat_treatment_df['Location'] == site]['projectid'].values[0]
                biomass_W_in = wheat_target_df.loc[wheat_target_df['projectid'] == simulation_id].loc[wheat_target_df['CURRENT.DATE'] == self.time_in]['AGBG_1_t_ha'].values[0]
                biomass_B_in = wheat_target_df.loc[wheat_target_df['projectid'] == simulation_id].loc[wheat_target_df['CURRENT.DATE'] == self.time_in]['AGBG_2_t_ha'].values[0]
                biomass_W_out = wheat_target_df.loc[wheat_target_df['projectid'] == simulation_id].loc[wheat_target_df['CURRENT.DATE'] == self.time_out]['AGBG_1_t_ha'].values[0]
                biomass_B_out = wheat_target_df.loc[wheat_target_df['projectid'] == simulation_id].loc[wheat_target_df['CURRENT.DATE'] == self.time_out]['AGBG_2_t_ha'].values[0]
            target_in = torch.tensor((biomass_B_in,biomass_W_in)).float()
            target_out = torch.tensor((biomass_B_out,biomass_W_out)).float()

            # # transform target
            if target_type == 'ratio':
                target_in = get_ratio_from_total_target(target_in)
                target_out = get_ratio_from_total_target(target_out)
            if target_transform is not None:
                target_in = torch.div(target_in, target_transform)
                target_out = torch.div(target_out, target_transform)
                
            self.all_targets_in.append(target_in)
            self.all_targets_out.append(target_out)
            
            # # img data
            if self.in_memory:
                self.data_in.append(init_transform(self.image_loader(in_path)))
                self.data_out.append(init_transform(self.image_loader(out_path)))
            
    def __len__(self):
        return len(self.all_in_paths)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # # access or load images
        if self.in_memory:
            image_1 = self.data_in[idx]
            image_2 = self.data_out[idx]
        else:
            image_1 = self.image_loader(self.all_in_paths[idx])
            image_2 = self.image_loader(self.all_out_paths[idx])
        
        # # transform both imgs (attention: set seed to transform both imgs in the same way)
        seed = np.random.randint(245985462)
        if self.transform:
            torch.manual_seed(seed)
            random.seed(seed)
            image_1 = self.transform(image_1)
            torch.manual_seed(seed)
            random.seed(seed)
            image_2 = self.transform(image_2)
        
        # # build sample dict
        sample = {'id': self.all_id[idx],
                  'img_1': image_1,
                  'name_1': self.all_names[idx],   
                  'time_1': self.time_in, 
                  'biomass_1': self.all_targets_in[idx],
                  'img_2': image_2,
                  'name_2': self.all_names[idx],   
                  'time_2': self.time_out,
                  'biomass_2': self.all_targets_out[idx],
                  'plotNr': self.all_plotNr[idx],
                  'orig_trt': self.all_orig_trt[idx],
                  'orig_dense': self.all_orig_dense[idx],
                  'actualMix': self.all_actualMix[idx],
                  'label': self.all_label[idx],
                  }
        
        return sample