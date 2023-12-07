import sys, os, warnings, time, logging
import yaml
import numpy as np
import math
import matplotlib
import torch

from datetime import datetime
from utils import utils

'''
Config for pred_transgrow.py as dictionary 'cfg'.
1. [REQUIRED] Specify exp_name and ckpt for which you want to run predictions
2. [AUTOMATIC] cfg_main.yaml is loaded from exp_name
3. [OPTIONAL] Update cfg as desired
4. [AUTOMATIC] Create pred_dir 
5. [OPTIONAL] Set font for figures

# *****************************************************************************
Config elements framed with # **** are dependent on other elements
# *****************************************************************************
'''


#%% 1. [REQUIRED] Specify exp_name and ckpt for which you want to have predictions
'''
log_dir : str
    directory, where the experiments are saved
exp_name : str
    name of exp to be loaded
ckpt_type
    which saved model ckpt should be used for prediction?
    choose e.g. 'last', 'best'
    Take care, if multiple 'best' epochs exist the last one is automatically chosen.
    For spcific epochs you should use e.g. 'best_epoch=123'
'''

log_dir = 'lightning_logs'
exp_name = '20230613_210941_img_t_if_mix_wgangp_img_256_z_128_t_64'
ckpt_type = 'best'
train_results = True


#%% 2. [AUTOMATIC] cfg_main.yaml (stored after training) is loaded from exp_name
'''
Load cfg_main.yml, which was stored in training and add/update the following cfg elements:
add/update log_dir, exp_name, ckpt_type, train_results, save_imgs (all described above)
add/update device, ckpt_path_pred

device : int
    current cuda device
ckpt_path_pred : str
    whole path of the spefified ckpt to run predictions from
'''

# *****************************************************************************
# # Load cfg from training
cfg_path = os.path.join(log_dir, exp_name, 'cfg_main.yaml')
with open(cfg_path, 'r') as stream:
    cfg = yaml.load(stream, Loader=yaml.Loader)
# # add/update parameters specified above
cfg.update({'log_dir': log_dir})
cfg.update({'exp_name': exp_name})
cfg.update({'device': torch.cuda.current_device()})
cfg['train_results'] = train_results 

# # Set ckpt path to run predictions from 
ckpt_dir = os.path.join(log_dir, exp_name, 'checkpoints')
ckpts_paths = utils.getListOfFiles(ckpt_dir)
ckpts_paths.sort(key=utils.natural_keys)
matching = [s for s in ckpts_paths if ckpt_type in s]
cfg['ckpt_path_pred'] = matching[-1]
cfg['ckpt_type'] = cfg['ckpt_path_pred'][cfg['ckpt_path_pred'].rindex('/')+1:-5]
# ***end***********************************************************************


#%% 3. [OPTIONAL] Update cfg as desired
'''
typically data params or transformations are varied, BUT NOT img_size

'''
cfg.update({'nworkers': 8})
cfg.update({'batch_size': 8})


#%% 4. [AUTOMATIC] Create pred_dir 
'''
pred_name : str
    Name of prediction folder
pred_dir : str
    directory of prediction folder in which all the prediction results are stored (in log_dir/exp_name)
'''

# *****************************************************************************
cfg['pred_name'] = datetime.now().strftime("%Y%m%d_%H%M%S")+'_pred_'+cfg['ckpt_type']
cfg['pred_dir'] = os.path.join(os.getcwd(), cfg['log_dir'], cfg['exp_name'], cfg['pred_name'])
if not os.path.exists(cfg['pred_dir']):
    os.makedirs(cfg['pred_dir'])
# ***end***********************************************************************


#%% 5. [OPTIONAL] Set font for figures
matplotlib.rcParams.update(
    {
    'text.usetex': True,
    "font.family": "serif",
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts" : False,
    }
)


