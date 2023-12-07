import sys, os
import numpy as np
import math
import matplotlib
import torchvision
import torch
import socket
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from datetime import datetime
from utils import utils
from utils.cutout import Shadowout
from utils.random_rot90 import RandomRot90


#%% Create config dict 'cfg' ==================================================
'''
Config for train_img_t_if.py as dictionary 'cfg'.
Find list and description of all elements in sections below.

# *****************************************************************************
Config elements framed with # **** are dependent on other elements
# *****************************************************************************
'''

cfg = {}


#%% GENERAL PARAMS ============================================================
'''
log_dir : str
    Name of folder where to store experiments
    Created in the project folder at the level of train_img_t_if_cls.py
c_type : str
    name of conditions separated with '_' (for filename)
    img ... image
    t ... time
    if ... influencing factors
    cls ... classes
resume_train_exp_name : str
    Specify the exp_name folder, if you want to resume training (need to be in folder log_dir)
    Specify None, if you don't want to resume training from a previous ckpt
    Attention: If you resume training ...
    ... you may need to increase the max_epochs, because it starts counting at the epoch where the old training stopped
    ... please ensure that data params and model params (see below) affecting the architecture (saved in hparams.yaml) are identical with config in this file, otherwise weight missmatch will occur
    DEFAULT: None
device : int
    which cuda device
'''

cfg['log_dir'] = 'lightning_logs'
cfg['c_type'] = 'img_t_if'
cfg['resume_train_exp_name'] = None
# cfg['resume_train_exp_name'] = '20230613_210941_img_t_if_mix_wgangp_img_256_z_128_t_64'

# ***start*********************************************************************
cfg['device'] = torch.cuda.current_device()
# ***end***********************************************************************


#%% DATA PARAMS ===============================================================
'''
data_name : str
    Name of dataset:
    'mix' = MixedCrop
    Attention: only 'mix' for all trainings implemented, as 'abdc' and 'grf' are missing the conditions 'cls' and 'if'
img_dir : str
    Path to img directory
data_time : dict contating datetime objects
    time_start : datetime.datetime object (observation start of dataset)
    time_end : datetime.datetime object (observation end of dataset)
    time_unit : string (modeling time unit, Choose from: 's'econd, 'm'inute, 'h'ours, 'd'ay or 'w'eek, The smaller the time unit, the larger the positional encoding vector will be)
info_tab_path : str
    Path to .csv where plot numbers are assigned to treatment information
wheat_target_path : str
    Path to .csv with daywise simulated dried biomass and height simulations for different wheat cultivars at different locations indexed with different "projectid" from the simulation software
bean_target_path : str
    Path to .csv with daywise simulated dried biomass and height simulations for different bean cultivars at different locations indexed with different "projectid" from the simulation software
mix_target_path : str
    Path to .csv with daywise simulated dried biomass and height simulations for different bean/wheat mixtures at different locations indexed with different "projectid" from the simulation software
'wheat_treatment_path'
    path to .csv with link from "projectid" of wheat simulations to the treatment
'bean_treatment_path'
    path to .csv with link from "projectid" of bean simulations to the treatment
'mix_treatment_path'
    path to .csv with link from "projectid" of mixture simulations to the treatment       
n_workers : int
    Number of workers, reasonable value depends on CPU kernels
    local typically 8, on GPU server 32 
batch_size : int
    Batch size, maximum possible number mainly depend on GPU memory
    local typically 8, on GPU server 64
in_memory : bool
    Specify, if all images should be loaded into the memory (in init of dataset class) [faster if memory large enough] or should be loaded in getItem (classic way)
img_ext : list of str
    Choose allowed img extension formats (e.g. 'png', 'jpg', 'tif')
img_size : int
    Processing img size
val_test_shuffle : bool
    shuffle val and test set in DataLoader?
normalize : string
    '01' normalize to [0 1]
    '11' normalize to [-1 1]
    'standardizing_abd' data set specific standardizing with precomputed mean and variance
    Attention: May change also cfg['final_actvn']
num_targets : int
    How many target parameter are used (e.g. =2 for wheat and bean biomass, more if you also load plant height -> modify dataset class)
target_type : str
    Relevant for biomass as target: Should bean/mix ratio targets ('ratio') be calculated or should total biomass targets be used ('total')?
target_transform : torch.tensor
    Tensor with size of num_targets with numbers by which target is divided (most likely to scale between [0 1])
    [max values for dried biomass in t/ha] torch.tensor((8.5,7)).float()
    Attention: Need to be the same shape as target (num_targets)
'''

cfg['data_name'] = 'mix'

# ***start*********************************************************************
if cfg['data_name'] == 'mix':
    cfg['img_dir'] = '../data/MixedCrop/Mix_RGB_CKA_2020/plantsort_patch_484/'
    cfg['data_time'] = {'time_start': datetime.strptime('2020-03-26', '%Y-%m-%d'), 
                        'time_end': datetime.strptime('2020-07-23', '%Y-%m-%d'),
                        'time_unit': 'd'}
    # # Data files with treatment information and dynamic process-based mixed crop biomass simulation: Dried biomass in t/ha
    cfg['info_tab_path'] = '../data/MixedCrop/CP5_2020_plotNo_treatment.csv'
    cfg['wheat_target_path'] = '../data/MixedCrop/Simulation/wheat.csv'
    cfg['bean_target_path'] = '../data/MixedCrop/Simulation/bean.csv'
    cfg['mix_target_path'] = '../data/MixedCrop/Simulation/mix.csv'
    cfg['wheat_treatment_path'] = '../data/MixedCrop/Simulation/treatment_information_wheat.csv'
    cfg['bean_treatment_path'] = '../data/MixedCrop/Simulation/treatment_information_bean.csv'
    cfg['mix_treatment_path'] = '../data/MixedCrop/Simulation/treatment_information_mix.csv'
# ***end***********************************************************************

if socket.gethostname() == 'lukas-P3630':
    cfg['n_workers'] = 8 
    cfg['batch_size'] = 8
else:
    cfg['n_workers'] = 32
    cfg['batch_size'] = 64
cfg['in_memory'] = False
cfg['img_ext'] = ['png', 'jpg', 'tif']
cfg['img_size'] = 256

cfg['val_test_shuffle'] = True
cfg['normalize'] = '01'

cfg['num_targets'] = 2
cfg['target_type'] = 'total'
cfg['target_transform'] = torch.tensor((8.5,7)).float()

# ***start*********************************************************************
# # overwrite target to transform to ones, if target type is ratio (then ratio is calculated inside dataset class and max is 1 anyway)
if cfg['target_type'] == 'ratio':
    cfg['target_transform'] = torch.ones(cfg['num_targets'])
    print('Target Transform overwritten to:', cfg['target_transform'])
# ***end***********************************************************************


#%% MODEL PARAMS ==============================================================
'''
use_model : str
    Which GAN model should be used? 'gan', 'wgan', 'wgangp'
    'wgangp': WGAN-GP
    DEFAULT = 'wgangp' (gan and wgan not for all trainings implemented)
g_e_net : str
    Genereator Encoder Network to produce img embeddings
    'res18': ResNet-18
    'res50': ResNet-50
    'res18_CBN': ResNet-18 with Conditional Batch Normalization (CBN) instead classic BN
    'res18_CBN_noPool': ResNet-18  with CBN instead BN and without last pooling layer (DEFAULT)
g_e_net_pretrained : bool
    Use pretrained weights?
    Only implemented for res18 and res50
g_d_net : str
    Generator Decoder Network to generate img from latent dimension
    'lightweight'
    'lightweight_CBN'
    'res18_CBN_noPool' (inverse of encoder)
g_d_net_pretrained : bool
    Use prertrained weights?
    not yet implemented, placeholder for future
dim_z : int
    dimension of stochasticity induced to the network
    DEFAULT = 16
dim_w : int
    dimension of mapped stochasticy using noise mapping similar to StyleGAN z->w
    if None -> no noise mapping
    if z_fusion_type='add'/'wadd' dim_w will be overwritten: dim_w=dim_img
dim_img : int
    embedding dimension of input image
dim_t : int
    input dimension of t (time) (after positional encoding)
dim_if : int
    input dimension of influencing factors (if)
z_fusion_type : str
    How are img embedding and z (or w if noise mapping is used) fused
    'add' : add z to img embedding
    'wadd' : weighted adding to to img embedding (depending on the time difference between target and input): high difference -> more noise
    'cat' : cat z to img embedding
t_fusion_type : str
    placeholder for future
    currently fused with CBN
d_net : str
    Which Discriminator Network? 
    'layer3_img_t_if'
    'layer5_img_t_if' 
d_net_pretrained : bool
    Use prertrained weights?
    not yet implemented
'''

cfg['use_model'] = 'wgangp'
cfg['g_e_net'] = 'res18_CBN_noPool'
cfg['g_e_net_pretrained'] = False
cfg['g_d_net'] = 'res18_CBN_noPool'
cfg['g_d_net_pretrained'] = False
cfg['dim_z'] = 128
cfg['dim_w'] = 512
cfg['dim_img'] = 512
cfg['dim_t'] = 64
cfg['dim_if'] = 2
cfg['z_fusion_type'] = 'add'
cfg['t_fusion_type'] = None

cfg['d_net'] = 'layer5_img_t_if'
cfg['d_net_pretrained'] = False

# ***start*********************************************************************
if cfg['dim_w'] is not None and (cfg['z_fusion_type']=='add' or cfg['z_fusion_type']=='wadd'):
    cfg['dim_w'] = cfg['dim_img']

if cfg['dim_w']==None and (cfg['z_fusion_type']=='add' or cfg['z_fusion_type']=='wadd'):
    cfg['dim_z'] = cfg['dim_img']
# ***end***********************************************************************


#%% OPTIMIZATION AND TRAINING PARAMS ==========================================
'''
lr : float
    learning rate
    DEFAULT: 1e-4
losses_w : dict
    Losses (str) and corresponding weights (int) to be used for training
    'adv' = adversarial loss
    'l1' = l1 distance (reconstruction) loss
    'ssim' = multi-scale structural similarity loss
    'lpips' = learned perceptual image patch similarity
    Attention: Comment out unused losses in model class for faster computation
    Attention: May check if activations are compatible beforehand
final_actvn : 'str'
    Coose from 'sigmoid', 'tanh', 'relu'
    Attention: normalization and losses may needs to be changes accordingly
    DEFAULT = 'sigmoid'
max_epochs : int
    Number of training epochs
    Attention: if you resume training this needs to be larger than before, otherwise it will do nothing
gpus : int
    Number of GPUs
    DEFAULT = 1
precision : int
    Precision mode, Choose 16 or 32
    DEFAULT = 32
fast_dev_run : bool
    only a trial trainings run?
    Default = False
limit_train_batches : float
    Limit train batches during training [0.0 1.0]
    DEFAULT: 1.0
    Attention: Set to 1.0 and NOT 1, if you want to include all train data, because otherwise only one training sample is taken
limit_val_batches : float
    Limit val batches during training [0.0 1.0]
limit_test_batches : float
    Limit test batches during training [0.0 1.0]
early_stop : bool
    Stop training, if val_loss does not decrease anymore
    DEFAULT: False, since sometimes tricky in GAN training
save_ckpts_last : bool
    Save last epoch?
    Default: True
save_ckpts_best : int
    How many top-k ckpts should be saved?
    This is measured by means of val_loss
    Default: 1
exp_name : 'str'
    Name of experiment folder in log_dir 
    Automatically build from start time of the experiment and some essential parameters
exp_dir : 'str'
    Experiment directory
ckpt_path_resume : str
    Ckpt from which the training resumes if resume_train_exp_name is not None
    Otherwise None
callbacks : list of pytorch_lightning.callbacks
    EarlyStopping, LearningRateMonitor, ModelCheckpoint
    Are created partly based on previous configs and partly based on other defaults (e.g. patience at EarlyStopping) that could be changed.
logger : pl.loggers
    Specify the logger, with which you want to log losses during training
    Default: wandb logger        
'''

cfg['lr'] = 1e-4
cfg['losses_w'] = {'weight_adv': 1,
                   'weight_l1': 0,
                   'weight_ssim': 0,
                   'weight_lpips': 0,}
cfg['final_actvn'] = 'sigmoid'
cfg['max_epochs'] = 5000
cfg['gpus'] = 1
cfg['precision'] = 32 
cfg['fast_dev_run'] = False
cfg['limit_train_batches'] = 1.00
cfg['limit_val_batches'] = 1.00
cfg['limit_test_batches'] = 1.00
cfg['early_stop'] = False
cfg['save_ckpts_last'] = True
cfg['save_ckpts_best'] = 3 

# ***start*********************************************************************
# # Exp Name
if cfg['resume_train_exp_name']:
    cfg['exp_name'] = cfg['resume_train_exp_name']
else:
    cfg['exp_name'] = datetime.now().strftime("%Y%m%d_%H%M%S")+'_'+cfg['c_type']+'_'+cfg['data_name']+'_'+cfg['use_model']+'_img_'+str(cfg['img_size'])+'_z_'+str(cfg['dim_z'])+'_t_'+str(cfg['dim_t'])
# # Exp directory
cfg['exp_dir'] = os.path.join(os.getcwd(), cfg['log_dir'], cfg['exp_name'])
if not os.path.exists(cfg['exp_dir']):
        os.makedirs(cfg['exp_dir'])
# # Ckpt_path_resume
if cfg['resume_train_exp_name']:
    cfg['ckpt_path_resume'] = cfg['exp_dir']+'/checkpoints/last.ckpt'
else:
    cfg['ckpt_path_resume'] = None

# # Callbacks
# # LearningRateMonitor
callbacks = []
callbacks = [
    LearningRateMonitor(log_momentum=True),
]
# # EarlyStopping
early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=0.005,
    mode="min",
    patience=5,
    verbose=True,
)
if cfg['early_stop']:
    callbacks = [early_stop_callback] + callbacks   
# # ModelChecker
model_checker = ModelCheckpoint(
    dirpath=os.path.join(cfg['log_dir'],cfg['exp_name'],'checkpoints'),
    monitor="val_loss",
    mode="min",
    save_last=cfg['save_ckpts_last'], 
    save_top_k=cfg['save_ckpts_best'], 
    filename="best_{epoch}",
)
if cfg['save_ckpts_best']:
    callbacks += [model_checker]
cfg['callbacks'] = callbacks

# # Logger
# # Tensorboard (but you need to modify code a bit (e.g. image logging, disable gradient watching) to get this running)
# cfg['logger'] = pl.loggers.TensorBoardLogger(save_dir=os.getcwd(),
#                                           version=cfg['exp_name'],
#                                           name=cfg['log_dir'])
# # Weights and Biases
cfg['logger'] = WandbLogger(name=cfg['exp_name'],project='cgan')
# ***end***********************************************************************

#%% TESTING AND PLOTTING PARAMS ===============================================
'''
run_test : bool
    Compute losses for test data?
    Default: False
run_plots : bool
    run some plots after training for direct visualization?
    Default: True
figure_width / figure_height : float
    specify figure width and height for saving of plots in optimal shape
plot_dpi : int
    specify dpi for optimal plotting resolution of figures 
'''

cfg['run_test'] = False
cfg['run_plots'] = True

matplotlib.rcParams.update(
    {
    'text.usetex': True,
    "font.family": "serif",
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts" : False,
    }
)
# factor points to inch
pt_to_in = 1/72.27

# this needs to be set to the desired plot width in points
# for example can be determined using '\showthe\textwidth' or 
# '\the\columnwidth' in latex
# typical columnwidth for two-column paper: 252
# typical textwidth for two-column paper: 516
figure_width_in = 252

figure_width = figure_width_in * pt_to_in
cfg['figure_width'] = np.round(figure_width, 2)
cfg['figure_height'] = 2.5

cfg['plot_dpi'] = 200 # if > 100, view size >> true image size
   
    
#%% TRANSFORMATION/AUGMENTATION PARAMS ========================================
'''
norm_mean : np.array
    normalization mean (used in torchvision.transforms.Normalize)
norm_std : np.array
    normalization standard deviation (used in torchvision.transforms.Normalize)
transform_train : torchvision.transforms.transforms.Compose
    Object of training transformations
    Note: Transformation depend on dataset
transform_test : torchvision.transforms.transforms.Compose
    Object of val/test transformations
d_transform : list
    List of Augmentations before feeding images into discriminator (do not compose!)
    Attentiion: Can slow down the training time drastically
p_d_transform : float
    Probability of each augmentation in the list to be applied
    Attentiion: Can slow down the training time drastically
    Discriminator augmentations have not proven to be useful if an image is available as a condition that can be augmented classically
deNorm : utils.utils.DeNormalize
    Function to deNormalize the Normalized img in the end (used primarly in plotting)
toPIL : torchvision.transforms.ToPILImage
    Function to generate PIL img from tensor (used primarly in plotting)
'''

# ***start*********************************************************************
if cfg['normalize']=='01':
    # # transform [0 1] to [0 1] -> mean=0, std=1 -> do nothing
    cfg['norm_mean'] = np.zeros(3)
    cfg['norm_std'] = np.ones(3)
elif cfg['normalize']=='11':
    # # shift [0 1] to [-1 1] (you could also do *2-1)
    cfg['norm_mean'] = np.asarray([0.5, 0.5, 0.5])
    cfg['norm_std'] = np.asarray([0.5, 0.5, 0.5])
elif cfg['normalize']=='standardizing_abd':
    # # normalize from [0 1] to mean 0 +- std -> mean = true mean of dataset, std= true std of dataset
    # # norm stats for Arabidopsis images
    cfg['norm_mean'] = np.asarray([0.32494438, 0.31354403, 0.26689884])
    cfg['norm_std'] = np.asarray([0.16464259, 0.16859856, 0.15636043])
else:
    print('ERROR: Wrong Normalization/Scaling Method specified.')
# ***end***********************************************************************

transforms = [
    torchvision.transforms.ToTensor(), # ToTensor transforms to [0 1], if input is uint8 PIL image, otherwise there is no transformation
    torchvision.transforms.Normalize(cfg['norm_mean'], cfg['norm_std']),
    torchvision.transforms.Resize(size=(cfg['img_size'],cfg['img_size']), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomVerticalFlip(p=0.5),
    # torchvision.transforms.RandomRotation(90, interpolation=torchvision.transforms.InterpolationMode.BILINEAR), 
    RandomRot90(dims=[1,2]),
    torchvision.transforms.RandomAffine(degrees=0,translate=(0.1,0.1),interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
    # torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),#, hue=0.2),
    Shadowout(5,25,50),
    ]

cfg['transform_train'] = torchvision.transforms.Compose(transforms)


transforms = [
    torchvision.transforms.ToTensor(), # ToTensor transforms to [0 1], if input is uint8 PIL image, otherwise there is no transformation
    torchvision.transforms.Normalize(cfg['norm_mean'], cfg['norm_std']),
    torchvision.transforms.Resize(size=(cfg['img_size'],cfg['img_size']), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True),
    ]

cfg['transform_test'] = torchvision.transforms.Compose(transforms)

# # Discriminator augmentations
d_transforms = [
    # torchvision.transforms.RandomHorizontalFlip(p=0.5),
    # torchvision.transforms.RandomVerticalFlip(p=0.5),
    # RandomRot90(dims=[1,2]),
    # torchvision.transforms.RandomAffine(degrees=0,translate=(0.1,0.1),interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
    # torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    # Shadowout(5,25,50),
    ]
# # initial probability for discriminator_transforms
cfg['d_transforms'] = torchvision.transforms.Compose(d_transforms)
cfg['p_d_transforms'] = 0 # have not proven to be useful if an image is available as a condition that can be augmented classically

# # denorrmalization
cfg['deNorm'] = utils.DeNormalize(cfg['norm_mean'], cfg['norm_std'])
cfg['toPIL'] = torchvision.transforms.ToPILImage()
