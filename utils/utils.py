'''
===============================================================================
Utils
===============================================================================
'''

import torch
from torchvision.utils import make_grid
import numpy as np
import os
import re

# =============================================================================
# generate a grid of video frames

def make_video_grid(x):
 
    x = x.clone().cpu()
    grid = torch.FloatTensor(x.size(0)*x.size(2), x.size(1), x.size(3), x.size(4)).fill_(1)
    k = 0
    for i in range(x.size(0)):
        for j in range(x.size(2)):            
            grid[k].copy_(x[i,:,j,:,:])
            k = k+1
    grid = make_grid(grid, nrow=x.size(2), padding=0, normalize=True, scale_each=False)
    return grid


# =============================================================================
# save a grid of video frames
    
def save_video_grid(x, path, imsize=512):
    
    from PIL import Image    
    grid = make_video_grid(x)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    if x.size(2)<x.size(0):
        imsize_ratio = int(x.size(0)/x.size(2))
        im = im.resize((imsize,int(imsize*imsize_ratio)), Image.NEAREST)   
    else:
        imsize_ratio = int(x.size(2)/x.size(0))
        im = im.resize((int(imsize*imsize_ratio), imsize), Image.NEAREST)      
    im.save(path)
    
    
# =============================================================================
# generate a grid of images

def make_image_grid(x, ngrid):
    
    x = x.clone().cpu()
    if pow(ngrid,2) < x.size(0):
        grid = make_grid(x[:ngrid*ngrid], nrow=ngrid, padding=0, normalize=True, scale_each=False)
    else:
        grid = torch.FloatTensor(ngrid*ngrid, x.size(1), x.size(2), x.size(3)).fill_(1)
        grid[:x.size(0)].copy_(x)
        grid = make_grid(grid, nrow=ngrid, padding=0, normalize=True, scale_each=False)
    return grid


# =============================================================================
# get a grid of images
    
def get_image_grid(x, imsize=512, ngrid=4, color='', size=2):
    
    from PIL import Image
    grid = make_image_grid(x, ngrid)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im = im.resize((imsize,imsize), Image.NEAREST)
    if color != '': 
        im = add_border(im, color, size)
    im = np.array(im)
    return im
    
    
# =============================================================================
# save a grid of images
    
def save_image_grid(x, path, imsize=512, ngrid=4, color='', size=2):
    
    from PIL import Image
    grid = make_image_grid(x, ngrid)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im = im.resize((imsize,imsize), Image.NEAREST)
    if color != '': 
        im = add_border(im, border=size, fill=color)
    im.save(path)


# =============================================================================
# add colored border to image
    
def add_border(x, color='', size=2):
    
    from PIL import ImageOps
    if color != '': 
        x = ImageOps.expand(x, border=size, fill=color)
    return x
        

# =============================================================================
# calculate output dimension of convolution
    
def get_out_dim_conv(dim, k, stride, pad):
    x = ((dim+2*pad-1*(k-1)-1)/stride)+1
    return x


# =============================================================================
# calculate output dimension of transposed convolution
    
def get_out_dim_conv_transpose(dim, k, stride, pad):
    x = (dim-1)*stride-2*pad+k
    return x


# =============================================================================
# count model parameters
    
def count_model_params(model):
    
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# create an Bunch object x out of dictionary d: so you can access (read) its values with the syntax x.foo instead of the clumsier d['foo']
# Note: the other way around would be d = vars(x)
class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)


# =============================================================================
# Denormalization class
class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        deNorm_tensor = tensor.clone()
        for t, m, s in zip(deNorm_tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return deNorm_tensor
    
# =============================================================================
def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles

# =============================================================================
def getListOfImgFiles(dirName):
    # run getListOfFiles
    allFiles = getListOfFiles(dirName)
    
    # filter img data
    for fichier in allFiles[:]: # im_names[:] makes a copy of im_names.
        if not(fichier.endswith(".png") or fichier.endswith(".jpg") or fichier.endswith(".tif")):
            allFiles.remove(fichier)
    
    return allFiles


# =============================================================================
# funcion by Immanuel 
def to_3channel(image):
    return image.repeat((3, 1, 1))


# =============================================================================
def allIdxOfElementInList(list, element):
    idx = []
    for i in range(len(list)):
        if list[i] == element:
            idx.append(i)
    return idx


# =============================================================================
def firstIdxOfElementInList(list, element):
    idx = list.index(element)
    return idx


# =============================================================================    
def get_plant_mask(img, th=0.25):
    # RGB img required in channel first order and in range [0 1]
    rgbvi = ((img[1,:,:]*img[1,:,:])-(img[0,:,:]*img[2,:,:]))/((img[1,:,:]*img[1,:,:])+(img[0,:,:]*img[2,:,:]))
    rgbvi[rgbvi<th]=0
    rgbvi[rgbvi>=th]=1
    return rgbvi


# ============================================================================= 
def pla_per_img(img, th=0.25):
    # RGB img required in channel first order and in range [0 1]
    rgbvi = get_plant_mask(img, th)
    return round((torch.sum(rgbvi)/(rgbvi.shape[0]*rgbvi.shape[1])).item(),4)


# ============================================================================= 
def calculate_psnr(img1, img2, max_value=255):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))


# ============================================================================= 
def get_seconds_factor_to_time_unit(time_unit):
    '''
    Function to get the factor (in seconds) to a given time unit
    
    Args: 
        time_unit: 's', 'm', 'h', 'd', or 'w' [str]
    
    Returns: 
        factor_to_unit: [int]
    '''
    if time_unit == 's':
        factor_to_unit = 1
    elif time_unit == 'm':
        factor_to_unit = 60
    elif time_unit == 'h':
        factor_to_unit = 3600
    elif time_unit == 'd':
        factor_to_unit = 86400
    elif time_unit == 'w':
        factor_to_unit = 604800
    else:
        print('Wrong time_unit specified.')
    return factor_to_unit


# ============================================================================= 
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


# ============================================================================= 
def make_folder(directory,name):
    """ generates a folder if not exists """
    folder = os.path.join(directory,name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder
