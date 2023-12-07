import sys, os, warnings, time, logging
import yaml
import numpy as np
import math
import matplotlib
import torch

from datetime import datetime
from utils import utils

'''
Config for pred_transgrow.py as dictionary 'cfg'
1. [REQUIRED] Specify exp_name and ckpt for which you want to run predictions
2. [AUTOMATIC] cfg_main.yaml is loaded from exp_name
3. [AUTOMATIC] Eval model configs
4. [OPTIONAL] Update cfg as desired
5. [AUTOMATIC] Create pred_dir 
6. [OPTIONAL] Set font for figures

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
exp_name = '20231204_201239_img_t_cls_mix_wgangp_img_256_z_128_t_64'
change_dataset = None #'mix-wg'
ckpt_type = 'best'
# # for paper plot
# date_in = '2020_04_23' # # for CKA 2-date simulations (28 to 54 DAS)
# date_out = '2020_05_19'
date_in = '2020_04_23' # # for CKA 2-date simulations (28 to 99 DAS) 
date_out = '2020_07_03'
# date_in = '2020_04_02' # # for CKA 2-date simulations (7 to 82 DAS)
# date_out = '2020_06_16'

train_results = False
in_memory = False


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
cfg.update({'in_memory': in_memory})
cfg['date_in'] = date_in
cfg['date_out'] = date_out
cfg['train_results'] = train_results
 

# # Set ckpt path to run predictions from 
ckpt_dir = os.path.join(log_dir, exp_name, 'checkpoints')
ckpts_paths = utils.getListOfFiles(ckpt_dir)
ckpts_paths.sort(key=utils.natural_keys)
matching = [s for s in ckpts_paths if ckpt_type in s]
cfg['ckpt_path_pred'] = matching[-1]
cfg['ckpt_type'] = cfg['ckpt_path_pred'][cfg['ckpt_path_pred'].rindex('/')+1:-5]
# ***end***********************************************************************

# ***start*********************************************************************
# # update dataset?
if change_dataset:
    cfg.update({'data_name': change_dataset})
    if cfg['data_name'] == 'mix-wg':
        cfg.update({'img_dir': '../data/MixedCrop/Mix_RGB_WG_2020/plantsort_patch_484/'})
        cfg.update({'data_time': {'time_start': datetime.strptime('2020-03-29', '%Y-%m-%d'), 
                                  'time_end': datetime.strptime('2020-07-20', '%Y-%m-%d'),
                                  'time_unit': 'd'}})
# ***end***********************************************************************


#%% 3. [AUTOMATIC] Eval model configs
'''
Set eval model weights in dependence of the dataset
'''
if cfg['data_name'] == 'mix':
    cfg['evalM_path'] = 'eval_model_weights/Mixed-CKA/20230816_110245_mix_time_specific_res18_256_total_relu_mse/checkpoints/best_epoch=2783.ckpt'
    cfg['evalM_target_transform'] = torch.tensor((8.5,7)).float()
    cfg['site'] = 'CKA'
elif cfg['data_name'] == 'mix-wg':
    cfg['evalM_path'] = 'eval_model_weights/Mixed-CKA/20230816_110245_mix_time_specific_res18_256_total_relu_mse/checkpoints/best_epoch=2783.ckpt'
    cfg['evalM_target_transform'] = torch.tensor((8.5,7)).float()
    cfg['site'] = 'WG'
    

#%% 4. [OPTIONAL] Update cfg as desired
'''
typically data params or transformations are varied, BUT NOT img_size

'''
cfg.update({'nworkers': 8})
cfg.update({'batch_size': 8})


#%% 5. [AUTOMATIC] Create pred_dir 
'''
pred_name : str
    Name of prediction folder
pred_dir : str
    directory of prediction folder in which all the prediction results are stored (in log_dir/exp_name)
'''

# *****************************************************************************
if change_dataset:
    cfg['pred_name'] = datetime.now().strftime("%Y%m%d_%H%M%S")+'_pred_'+cfg['ckpt_type']+'_test_'+cfg['data_name']
else:
    cfg['pred_name'] = datetime.now().strftime("%Y%m%d_%H%M%S")+'_pred_'+cfg['ckpt_type']      
cfg['pred_dir'] = os.path.join(os.getcwd(), cfg['log_dir'], cfg['exp_name'], cfg['pred_name'])
if not os.path.exists(cfg['pred_dir']):
    os.makedirs(cfg['pred_dir'])
# ***end***********************************************************************


#%% 6. [OPTIONAL] Set font for figures
matplotlib.rcParams.update(
    {
    'text.usetex': True,
    "font.family": "serif",
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts" : False,
    }
)
