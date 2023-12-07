#from __future__ import print_function, division
import torch
import os
import re
import numpy as np
import datetime
import math
import random
import torchvision

from PIL import Image
from torch.utils.data import Dataset


from utils import utils


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
    
    if data_name == 'abd':
        time_string = img_path[-32:-12]
        time_obj = datetime.datetime.strptime(time_string, '%Y-%m-%d--%H-%M-%S')
    elif data_name == 'abdc':
        time_string = img_path[-42:-22]
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


def convert_datestr(date, data_name):
    '''
    Convert date str to another date str which fit to the dataset style
    
    Args:
        date: str
            Date str in format '%Y_%m_%d'
        data_name :  str
            name of dataset
    
    Return
        converted_date
            Date str in format given in specific dataset
    '''
    
    if data_name == 'abd' or data_name == 'abdc':
        converted_date = date.replace("_","-")
    else:
        converted_date = date      
        
    return converted_date


class RGBPlantImageDataset(Dataset):
    def __init__(self, rgb_dir, data_name, data_time, transform, in_memory=True):
        self.transform = transform
        self.in_memory = in_memory
        self.image_loader = default_image_loader()

        factor_to_unit = utils.get_seconds_factor_to_time_unit(data_time['time_unit'])

        # # paths, names and idx
        seq_names, seq_idx = find_seqs(rgb_dir)
        self.all_paths = []
        self.all_names = []
        self.all_id = []
        for name in seq_names:
            plant_paths = utils.getListOfImgFiles(rgb_dir+name)
            plant_names = [name] * len(plant_paths)
            plant_id = [seq_idx[name]] * len(plant_paths)
            self.all_paths += plant_paths
            self.all_names += plant_names
            self.all_id += plant_id
            
        # # times and data to memory
        self.all_times=[]
        if self.in_memory:
            self.data = []
            # # split transform as the first three (ToTensor, Normalize, and Resize) can be applied in init if self.in_memory
            init_transform = []
            getItem_transform = []
            for idx, transforms in enumerate(self.transform.transforms):
                if idx<=2:
                    init_transform.append(transforms)
                else:    
                    getItem_transform.append(transforms)
            init_transform = torchvision.transforms.Compose(init_transform)
            self.transform = torchvision.transforms.Compose(getItem_transform)
            
        for path in self.all_paths:
            # # times
            time = get_time_from_path(path, data_name)
            timedelta_ToStart = time-data_time['time_start'] # timedelta object
            timedelta = round(timedelta_ToStart.total_seconds()/factor_to_unit) # int
            self.all_times+=[timedelta]
            # # data
            if self.in_memory:
                self.data.append(init_transform(self.image_loader(path)))
    
    def __len__(self):
        return len(self.all_paths)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # # access or load images
        if self.in_memory:
            image_1 = self.data[idx]
        else:
            image_1 = self.image_loader(self.all_paths[idx])
        
        # # transform
        if self.transform:
            image_1 = self.transform(image_1)
        
        # # build sample dict
        sample = {'id': self.all_id[idx],
                  'img': image_1,
                  'name': self.all_names[idx],   
                  'time': self.all_times[idx], 
                  }
        
        return sample        
    
    
    
class RGBPlant2ImagesDataset(Dataset):
    def __init__(self, rgb_dir, data_name, data_time, transform, in_memory=True):
        self.transform = transform
        self.in_memory = in_memory
        self.image_loader = default_image_loader()
        
        factor_to_unit = utils.get_seconds_factor_to_time_unit(data_time['time_unit'])

        # # paths, names and idx
        seq_names, seq_idx = find_seqs(rgb_dir)
        self.all_paths = []
        self.all_names = []
        self.all_id = []
        for name in seq_names:
            plant_paths = utils.getListOfImgFiles(rgb_dir+name)
            plant_names = [name] * len(plant_paths)
            plant_id = [seq_idx[name]] * len(plant_paths)
            self.all_paths += plant_paths
            self.all_names += plant_names
            self.all_id += plant_id
            
        # # times and data to memory
        self.all_times=[]
        if self.in_memory:
            self.data = []
            # # split transform as the first three (ToTensor, Normalize, and Resize) can be applied in init if self.in_memory
            init_transform = []
            getItem_transform = []
            for idx, transforms in enumerate(self.transform.transforms):
                if idx<=2:
                    init_transform.append(transforms)
                else:    
                    getItem_transform.append(transforms)
            init_transform = torchvision.transforms.Compose(init_transform)
            self.transform = torchvision.transforms.Compose(getItem_transform)
            
        for path in self.all_paths:
            # # times
            time = get_time_from_path(path, data_name)
            timedelta_ToStart = time-data_time['time_start'] # timedelta object
            timedelta = round(timedelta_ToStart.total_seconds()/factor_to_unit) # int
            self.all_times+=[timedelta]
            # # data
            if self.in_memory:
                self.data.append(init_transform(self.image_loader(path)))
        
        self.plant_seq_assignment = []
        for plant_id in self.all_id:
            other_idx = np.where(np.array(self.all_id)==plant_id)[0]
            self.plant_seq_assignment.append(other_idx)
    
    def __len__(self):
        return len(self.all_paths)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # get random other idx of the same plant
        rand_idx = random.randint(0,len(self.plant_seq_assignment[idx])-1)
        idx2 = self.plant_seq_assignment[idx][rand_idx]
        
        # # access or load images
        if self.in_memory:
            image_1 = self.data[idx]
            image_2 = self.data[idx2]
        else:
            image_1 = self.image_loader(self.all_paths[idx])
            image_2 = self.image_loader(self.all_paths[idx2])
        
        # # transform both imgs (attention: transform will be differently for random augmentations -> set a seed
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
                  'img_2': image_2,
                  'name_2': self.all_names[idx2],   
                  'time_2': self.all_times[idx2],
                  }
        
        return sample    



class RGBPlant2Images2DatesDataset(Dataset):
    def __init__(self, img_dir, data_name, data_time, date_in, date_out, transform, in_memory=True):
        self.transform = transform
        self.in_memory = in_memory
        self.image_loader = default_image_loader()
        
        # # # # 
        factor_to_unit = utils.get_seconds_factor_to_time_unit(data_time['time_unit'])

        # # paths, names and idx
        seq_names, seq_idx = find_seqs(img_dir)
        self.in_plant_paths = []
        self.out_plant_paths = []
        self.plant_name = []
        self.plant_plotNr = []
        self.plant_id = []
        for name in seq_names:
            plant_paths = utils.getListOfImgFiles(img_dir+name)
            plant_name = name
            plant_id = seq_idx[name]
            filter_in_idx = [idx for idx, item in enumerate(plant_paths) if convert_datestr(date_in,data_name) in item]
            filter_out_idx = [idx for idx, item in enumerate(plant_paths) if convert_datestr(date_out,data_name) in item]
            if filter_in_idx and filter_out_idx:
                self.in_plant_paths.append(plant_paths[filter_in_idx[0]])
                self.out_plant_paths.append(plant_paths[filter_out_idx[0]])
                self.plant_name.append(plant_name)
                self.plant_id.append(plant_id)
        
        time = datetime.datetime.strptime(date_in, '%Y_%m_%d')
        timedelta_ToStart = time-data_time['time_start'] # timedelta object
        self.time_in = round(timedelta_ToStart.total_seconds()/factor_to_unit) # int
        
        time = datetime.datetime.strptime(date_out, '%Y_%m_%d')
        timedelta_ToStart = time-data_time['time_start'] # timedelta object
        self.time_out = round(timedelta_ToStart.total_seconds()/factor_to_unit) # int
        
        if self.in_memory:
            self.data_in = []
            self.data_out = []
            # # split transform as the first three (ToTensor, Normalize, and Resize) can be applied in init if self.in_memory
            init_transform = []
            getItem_transform = []
            for idx, transforms in enumerate(self.transform.transforms):
                if idx<=2:
                    init_transform.append(transforms)
                else:    
                    getItem_transform.append(transforms)
            init_transform = torchvision.transforms.Compose(init_transform)
            self.transform = torchvision.transforms.Compose(getItem_transform)
        
            for (in_path,out_path) in zip(self.in_plant_paths,self.out_plant_paths):
                self.data_in.append(init_transform(self.image_loader(in_path)))
                self.data_out.append(init_transform(self.image_loader(out_path)))
    
    def __len__(self):
        return len(self.in_plant_paths)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
       
        # # access or load images
        if self.in_memory:
            image_1 = self.data_in[idx]
            image_2 = self.data_out[idx]
        else:
            image_1 = self.image_loader(self.in_plant_paths[idx])
            image_2 = self.image_loader(self.out_plant_paths[idx])
        
        # # transform both imgs (attention: transform will be differently for random augmentations -> set a seed
        seed = np.random.randint(245985462)
        if self.transform:
            torch.manual_seed(seed)
            random.seed(seed)
            image_1 = self.transform(image_1)
            torch.manual_seed(seed)
            random.seed(seed)
            image_2 = self.transform(image_2)
        
        # # build sample dict
        sample = {'id': self.plant_id[idx],
                  'img_1': image_1,
                  'name_1': self.plant_name[idx],
                  'time_1': self.time_in,
                  'img_2': image_2,
                  'name_2': self.plant_name[idx],   
                  'time_2': self.time_out,
                  }
        
        return sample


    
# # only for visualizing purposes
class RGBPlantSeqDataset(Dataset):
    def __init__(self, rgb_dir, data_name, data_time, transform):
        self.transform = transform
        self.image_loader = default_image_loader()
        
        factor_to_unit = utils.get_seconds_factor_to_time_unit(data_time['time_unit'])

        # # paths, names and idx
        seq_names, seq_idx = find_seqs(rgb_dir)
        self.data_dict = {}
        for name in seq_names:
            plant_paths = sorted(utils.getListOfImgFiles(rgb_dir+name))
            plant_id = [seq_idx[name]] * len(plant_paths)
            self.data_dict[str(plant_id[0])] = {}
            self.data_dict[str(plant_id[0])]['plant_name'] = name
            self.data_dict[str(plant_id[0])]['plant_paths'] = plant_paths
        
        # # times
        for key in self.data_dict.keys():
            plant_paths = self.data_dict[key]['plant_paths']
            plant_times = []
            for path in plant_paths:
                time = get_time_from_path(path, data_name)
                timedelta_ToStart = time-data_time['time_start'] # timedelta object
                timedelta = round(timedelta_ToStart.total_seconds()/factor_to_unit) # int
                plant_times+=[timedelta]
            self.data_dict[key]['plant_times'] = plant_times
    
    def __len__(self):
        return len(self.data_dict.keys())
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # # load all images of seq
        seed = np.random.randint(245985462)
        all_imgs = []
        for path in self.data_dict[str(idx)]['plant_paths']:
            img = self.image_loader(path)
            if self.transform:
                torch.manual_seed(seed)
                random.seed(seed)
                img = self.transform(img)    
            all_imgs.append(img)
            
        imgs = torch.stack(all_imgs, dim=0)
            
        
        sample = {'id': idx,
                  'name': self.data_dict[str(idx)]['plant_name'],   
                  'img': imgs,
                  'time': self.data_dict[str(idx)]['plant_times'], 
                  }
        
        return sample