"""
===============================================================================
Get predictions and plots for models coming from train_img_t.py
===============================================================================
"""

import sys, os, io, warnings, time, logging
import torch
import torch.multiprocessing
import pytorch_lightning as pl
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import random
import yaml
import math
from utils import utils
from utils import evaluate_pla
from PIL import Image

from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance

from configs.config_test_img_t import cfg

from datasets.rgb_plant_image_dataset import RGBPlantSeqDataset
from datasets.rgb_plant_image_datamodule import RGBPlant2ImagesDataModule
from datasets.mc_datamodule import MixCropDataModule, MixCrop2ImagesDataModule, MixCrop2Images2DatesDataModule
from models.gan_img_t_plm import GANModel_img_t
from models.wgangp_img_t_plm import WGANGPModel_img_t

from eval_models.get_instance_segmentation_model import get_instance_segmentation_model
from eval_models.mc_time_specific_plm import MixCropTimeSpecificModel


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# # this makes lightning reports not look like errors
pl._logger.handlers = [logging.StreamHandler(sys.stdout)]

# # this line can avoid bugs on gpu servers
torch.multiprocessing.set_sharing_strategy('file_system')


#%% print versions stuff
print('python', sys.version, sys.executable)
print('pytorch', torch.__version__)
print('torchvision', torchvision.__version__)
print('pytorch-lightning', pl.__version__)
print('CUDA Available:', torch.cuda.is_available())
print(torch._C._cuda_getCompiledVersion(), 'cuda compiled version')
print(torch._C._nccl_version(), 'nccl')
for i in range(torch.cuda.device_count()):
    print('device %s:'%i, torch.cuda.get_device_properties(i))
    

#%% 
if __name__ == '__main__':    
    #%% write cfg.yaml to pred_dir
    with io.open(os.path.join(cfg['pred_dir'], 'cfg_pred.yaml'), 'w', encoding='utf8') as outfile:
        yaml.dump(cfg, outfile, default_flow_style=False, allow_unicode=True)
    
    #%% dataModule
    if cfg['data_name'] == 'abdc' or cfg['data_name'] == 'grf':
        dataModule = RGBPlant2ImagesDataModule(cfg['batch_size'], cfg['n_workers'], cfg['img_dir'], cfg['data_name'], cfg['data_time'], cfg['transform_train'], cfg['transform_test'], in_memory=cfg['in_memory'], val_test_shuffle=cfg['val_test_shuffle'])
    elif cfg['data_name'] == 'mix' or cfg['data_name'] == 'mix-wg':
        dataModule = MixCrop2ImagesDataModule(cfg['img_dir'], cfg['info_tab_path'], cfg['wheat_target_path'], cfg['bean_target_path'], cfg['mix_target_path'], cfg['wheat_treatment_path'], cfg['bean_treatment_path'], cfg['mix_treatment_path'], cfg['data_name'], cfg['data_time'], cfg['batch_size'], cfg['n_workers'], cfg['transform_train'], cfg['transform_test'], target_type=cfg['target_type'], target_transform=cfg['target_transform'], in_memory=cfg['in_memory'], val_test_shuffle=cfg['val_test_shuffle'])
    
    eval_dataset = RGBPlantSeqDataset(cfg['img_dir']+'eval/', cfg['data_name'], cfg['data_time'], transform=cfg['transform_test'])

    # setup dataModule
    dataModule.prepare_data()
    dataModule.setup()
    
    # show dim and len of different data subsets
    print('---Some Training Stats---')
    print('Input dims:', dataModule.data_dims)
    print('#Traindata:', len(dataModule.train_dataloader().dataset))
    print('#Valdata:', len(dataModule.val_dataloader().dataset))
    print('#Testdata:', len(dataModule.test_dataloader().dataset))
    
    if cfg['train_results']:
        dataloader_list = [dataModule.test_dataloader(), dataModule.train_dataloader()]
        prfx=['test_','train_']
    else:
        dataloader_list = [dataModule.test_dataloader()]
        prfx=['test_']
        
    
    #%% load model from checkpoint        
    if cfg['use_model'] == 'gan':
        model = GANModel_img_t.load_from_checkpoint(cfg['ckpt_path_pred'])
    elif cfg['use_model'] == 'wgangp':
        model = WGANGPModel_img_t.load_from_checkpoint(cfg['ckpt_path_pred'])
        
    # # set to eval mode
    model.eval()     
    # # sent model to device
    model.to(cfg['device'])
    
    # # Losses
    loss_l1 = torch.nn.L1Loss(reduction='none')
    loss_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0, reduction=None)
    loss_lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True)
    
    
    #%% load eval model (Mask R-CNN or Regression model)
    if cfg['data_name'] == 'abdc' or cfg['data_name'] == 'grf':
        eval_model = get_instance_segmentation_model(cfg['evalM_num_classes'], cfg['evalM_version'])
        eval_model.load_state_dict(torch.load(cfg['evalM_path']))
    elif cfg['data_name'] == 'mix' or cfg['data_name'] == 'mix-wg':
        eval_model = MixCropTimeSpecificModel.load_from_checkpoint(cfg['evalM_path'])
    
    # move model to the right device
    eval_model.to(cfg['device'])
    eval_model.eval()
    
    print('Evaluation model:', cfg['evalM_path'], 'loaded successfully.')
        
    
    #%% start predicting / scoring / plotting 
    for count, dataloader in enumerate(dataloader_list):
        
        #%% calculate metrics
        fid = FrechetInceptionDistance(feature=2048).to(cfg['device'])
        t_gen = []
        t_diff = []
        l1 = []
        ssim = []
        lpips = []
        psnr = []
        pla_diff = []
        pla_diff_norm = []
        pla_ref = []
        biomass_diff = []
        biomass_ref = []
        label = []
        
        # # plot dir for scores
        plot_dir = utils.make_folder(cfg['pred_dir'],(prfx[count]+'metrics'))
        
        # # avoid going into the last batch as it is sometimes < batch_size and this causes problems later on
        max_batches = math.floor(len(dataloader.dataset)/cfg['batch_size'])
        # max_batches = 2
        for i_batch, batch in enumerate(dataloader):
            if i_batch==max_batches:
                break
            
            with torch.no_grad():
                # # img_in, t_in, if_in
                img_in = batch['img_1']
                t_in = batch['time_1']
                
                # # img_ref, t_ref, if_ref
                img_ref = batch['img_2'].to(cfg['device'])
                t_ref = batch['time_2']
                
                # # save times to list
                t_gen.append(t_ref)
                t_diff.append(t_ref-t_in)
    
                # # run model
                img_gen = model(img_in=img_in,t_in=t_in,t_ref=t_ref)
                
                # # calculate FID
                fid.update((cfg['deNorm'](img_ref)*255).to(torch.uint8), real=True)
                fid.update((cfg['deNorm'](img_gen)*255).to(torch.uint8), real=False)
                
                # # L1, SSIM, LPIPS, PSNR and EvalM-metric
                l1.append(torch.mean(loss_l1(img_gen, img_ref), [1,2,3]).cpu())
                ssim.append(loss_ssim(img_gen, img_ref).cpu())
                if cfg['data_name'] == 'abdc': # PLA in mm
                    pla_diff.append(evaluate_pla.loss_pla(eval_model(img_gen), eval_model(img_ref), threshold=cfg['pla_threshold'], center_threshold=cfg['pla_center_threshold'], normalize=False)*((cfg['GSD_cm']*10)**2))
                    pla_diff_norm.append(evaluate_pla.loss_pla(eval_model(img_gen), eval_model(img_ref), threshold=cfg['pla_threshold'], center_threshold=cfg['pla_center_threshold'], normalize=True))
                    pla_ref.append(evaluate_pla.get_pla(eval_model(img_ref), threshold=cfg['pla_threshold'], center_threshold=cfg['pla_center_threshold'], normalize=False)*((cfg['GSD_cm']*10)**2))
                elif cfg['data_name'] == 'grf': # PLA in cm
                    pla_diff.append(evaluate_pla.loss_pla(eval_model(img_gen), eval_model(img_ref), threshold=cfg['pla_threshold'], center_threshold=cfg['pla_center_threshold'], normalize=False)*(cfg['GSD_cm']**2))
                    pla_diff_norm.append(evaluate_pla.loss_pla(eval_model(img_gen), eval_model(img_ref), threshold=cfg['pla_threshold'], center_threshold=cfg['pla_center_threshold'], normalize=True))
                    pla_ref.append(evaluate_pla.get_pla(eval_model(img_ref), threshold=cfg['pla_threshold'], center_threshold=cfg['pla_center_threshold'], normalize=False)*(cfg['GSD_cm']**2))
                elif cfg['data_name'] == 'mix' or cfg['data_name']=='mix-wg':
                    biomass_diff.append(((eval_model(img_gen)-eval_model(img_ref))).cpu().detach()*cfg['evalM_target_transform'])
                    biomass_ref.append((eval_model(img_ref)).cpu().detach()*cfg['evalM_target_transform'])
                    label.append(batch['label'])
                for k in range(img_ref.shape[0]):
                    lpips.append(loss_lpips(img_gen[k,:].unsqueeze(dim=0).cpu().detach(), img_ref[k,:].unsqueeze(dim=0).cpu().detach()).item())
                    psnr.append(utils.calculate_psnr(img_gen[k,:].cpu().detach(),img_ref[k,:].cpu().detach(), max_value=1).item())
                
        t_gen = np.array(torch.cat(t_gen))
        t_diff = np.array(torch.cat(t_diff))
        t0_filter = np.where((abs(t_diff)==0))[0]
        t1_filter = np.where((abs(t_diff)>0) & (abs(t_diff)<=10))[0]
        t2_filter = np.where((abs(t_diff)>10))[0]
        l1 = np.array(torch.cat(l1))
        ssim = np.array(torch.cat(ssim))
        lpips = np.array(lpips)
        psnr = np.array(psnr)

        if cfg['data_name'] == 'abdc' or cfg['data_name'] == 'grf':
            pla_diff = np.array(torch.cat(pla_diff))
            pla_diff_norm = np.array(torch.cat(pla_diff_norm))
            pla_ref = np.array(torch.cat(pla_ref))
            # since we devide by pla_ref, it should not be zero
            pla_ref[pla_ref==0]=1
            scaled_delta_pla = (pla_diff/pla_ref)
            # remove outlier coming from dividing by very small numbers (trash masks)
            scaled_delta_pla[scaled_delta_pla<-10]=-10
            scaled_delta_pla[scaled_delta_pla>10]=10
            scores = {'FID': str(fid.compute().item()),
                      'L1': str(np.mean(l1)),
                      'L1 std': str(np.std(l1)),
                      'SSIM': str(np.mean(ssim)),
                      'SSIM std': str(np.std(ssim)),
                      
                      't0_filter SSIM': str(np.mean(ssim[t0_filter])),
                      't0_filter SSIM std': str(np.std(ssim[t0_filter])),
                      't1_filter SSIM': str(np.mean(ssim[t1_filter])),
                      't1_filter SSIM std': str(np.std(ssim[t1_filter])),
                      't2_filter SSIM': str(np.mean(ssim[t2_filter])),
                      't2_filter SSIM std': str(np.std(ssim[t2_filter])),
                      
                      'LPIPS': str(np.mean(lpips)),
                      'LPIPS std': str(np.std(lpips)),
                      
                      't0_filter LPIPS': str(np.mean(lpips[t0_filter])),
                      't0_filter LPIPS std': str(np.std(lpips[t0_filter])),
                      't1_filter LPIPS': str(np.mean(lpips[t1_filter])),
                      't1_filter LPIPS std': str(np.std(lpips[t1_filter])),
                      't2_filter LPIPS': str(np.mean(lpips[t2_filter])),
                      't2_filter LPIPS std': str(np.std(lpips[t2_filter])),
                      
                      'PSNR': str(np.mean(psnr)),
                      'PSNR std': str(np.std(psnr)),
                      'PLA ME': str(np.mean(pla_diff)),
                      'PLA ME std': str(np.std(pla_diff)),
                      'PLA MAE': str(np.mean(abs(pla_diff))),
                      'PLA MAE std': str(np.std(abs(pla_diff))),
                      
                      't0_filter PLA ME': str(np.mean(pla_diff[t0_filter])),
                      't0_filter PLA ME std': str(np.std(pla_diff[t0_filter])),
                      't0_filter PLA MAE': str(np.mean(abs(pla_diff[t0_filter]))),
                      't0_filter PLA MAE std': str(np.std(abs(pla_diff[t0_filter]))),
                      't1_filter PLA ME': str(np.mean(pla_diff[t1_filter])),
                      't1_filter PLA ME std': str(np.std(pla_diff[t1_filter])),
                      't1_filter PLA MAE': str(np.mean(abs(pla_diff[t1_filter]))),
                      't1_filter PLA MAE std': str(np.std(abs(pla_diff[t1_filter]))),
                      't2_filter PLA ME': str(np.mean(pla_diff[t2_filter])),
                      't2_filter PLA ME std': str(np.std(pla_diff[t2_filter])),
                      't2_filter PLA MAE': str(np.mean(abs(pla_diff[t2_filter]))),
                      't2_filter PLA MAE std': str(np.std(abs(pla_diff[t2_filter]))),
                      
                      'PLA norm ME': str(np.mean(pla_diff_norm)),
                      'PLA norm ME std': str(np.std(pla_diff_norm)),
                      'PLA norm MAE': str(np.mean(abs(pla_diff_norm))),
                      'PLA norm MAE std': str(np.std(abs(pla_diff_norm))),
                      
                      't0_filter PLA norm ME': str(np.mean(pla_diff_norm[t0_filter])),
                      't0_filter PLA norm ME std': str(np.std(pla_diff_norm[t0_filter])),
                      't0_filter PLA norm MAE': str(np.mean(abs(pla_diff_norm[t0_filter]))),
                      't0_filter PLA norm MAE std': str(np.std(abs(pla_diff_norm[t0_filter]))),
                      't1_filter PLA norm ME': str(np.mean(pla_diff_norm[t1_filter])),
                      't1_filter PLA norm ME std': str(np.std(pla_diff_norm[t1_filter])),
                      't1_filter PLA norm MAE': str(np.mean(abs(pla_diff_norm[t1_filter]))),
                      't1_filter PLA norm MAE std': str(np.std(abs(pla_diff_norm[t1_filter]))),
                      't2_filter PLA norm ME': str(np.mean(pla_diff_norm[t2_filter])),
                      't2_filter PLA norm ME std': str(np.std(pla_diff_norm[t2_filter])),
                      't2_filter PLA norm MAE': str(np.mean(abs(pla_diff_norm[t2_filter]))),
                      't2_filter PLA norm MAE std': str(np.std(abs(pla_diff_norm[t2_filter]))),
                      
                      # 'Scaled Delta PLA ((gen-ref)/ref)': str(np.mean(scaled_delta_pla)),
                      # 'Scaled Delta PLA ((gen-ref)/ref) std': str(np.std(scaled_delta_pla)),
                      # 'Scaled Delta PLA (abs((gen-ref)/ref))': str(np.mean(abs(scaled_delta_pla))),
                      # 'Scaled Delta PLA (abs((gen-ref)/ref)) std': str(np.std(abs(scaled_delta_pla))),
                      }
        elif cfg['data_name'] == 'mix' or cfg['data_name'] == 'mix-wg':
            biomass_diff = np.array(torch.stack(biomass_diff).view(-1,2))
            biomass_ref = np.array(torch.stack(biomass_ref).view(-1,2))
            label = np.array(torch.cat(label))
            mix_idx = np.where((label>=28) & (label<=75))[0]
            scores = {'FID': str(fid.compute().item()),
                      'L1': str(np.mean(l1)),
                      'L1 std': str(np.std(l1)),
                      'SSIM': str(np.mean(ssim)),
                      'SSIM std': str(np.std(ssim)),
                      
                      't0_filter SSIM': str(np.mean(ssim[t0_filter])),
                      't0_filter SSIM std': str(np.std(ssim[t0_filter])),
                      't1_filter SSIM': str(np.mean(ssim[t1_filter])),
                      't1_filter SSIM std': str(np.std(ssim[t1_filter])),
                      't2_filter SSIM': str(np.mean(ssim[t2_filter])),
                      't2_filter SSIM std': str(np.std(ssim[t2_filter])),
                      
                      'LPIPS': str(np.mean(lpips)),
                      'LPIPS std': str(np.std(lpips)),
                      
                      't0_filter LPIPS': str(np.mean(lpips[t0_filter])),
                      't0_filter LPIPS std': str(np.std(lpips[t0_filter])),
                      't1_filter LPIPS': str(np.mean(lpips[t1_filter])),
                      't1_filter LPIPS std': str(np.std(lpips[t1_filter])),
                      't2_filter LPIPS': str(np.mean(lpips[t2_filter])),
                      't2_filter LPIPS std': str(np.std(lpips[t2_filter])),
                      
                      'PSNR': str(np.mean(psnr)),
                      'PSNR std': str(np.std(psnr)),
                      'BM ME': str(np.mean(biomass_diff, axis=0)),
                      'BM ME std': str(np.std(biomass_diff, axis=0)),
                      'BM MAE': str(np.mean(abs(biomass_diff), axis=0)),
                      'BM MAE std': str(np.std(abs(biomass_diff), axis=0)),
                      
                      't0_filter BM ME': str(np.mean(biomass_diff[t0_filter], axis=0)),
                      't0_filter BM ME std': str(np.std(biomass_diff[t0_filter], axis=0)),
                      't0_filter BM MAE': str(np.mean(abs(biomass_diff[t0_filter]), axis=0)),
                      't0_filter BM MAE std': str(np.std(abs(biomass_diff[t0_filter]), axis=0)),
                      't1_filter BM ME': str(np.mean(biomass_diff[t1_filter], axis=0)),
                      't1_filter BM ME std': str(np.std(biomass_diff[t1_filter], axis=0)),
                      't1_filter BM MAE': str(np.mean(abs(biomass_diff[t1_filter]), axis=0)),
                      't1_filter BM MAE std': str(np.std(abs(biomass_diff[t1_filter]), axis=0)),
                      't2_filter BM ME': str(np.mean(biomass_diff[t2_filter], axis=0)),
                      't2_filter BM ME std': str(np.std(biomass_diff[t2_filter], axis=0)),
                      't2_filter BM MAE': str(np.mean(abs(biomass_diff[t2_filter]), axis=0)),
                      't2_filter BM MAE std': str(np.std(abs(biomass_diff[t2_filter]), axis=0)),
                      
                      'Mix BM ME': str(np.mean(biomass_diff[mix_idx], axis=0)),
                      'Mix BM ME std': str(np.std(biomass_diff[mix_idx], axis=0)),
                      'Mix BM MAE': str(np.mean(abs(biomass_diff[mix_idx]), axis=0)),
                      'Mix BM MAE std': str(np.std(abs(biomass_diff[mix_idx]), axis=0)),
                      }

        with open(os.path.join(plot_dir,'scores.yaml'), 'w') as file:
            yaml.dump(scores, file)   
        
    
        #%% generate random imgs
        max_plots = 5
        n_imgs = 8 # per plot, needs to be <= batch_size
        plot_dir = utils.make_folder(cfg['pred_dir'],(prfx[count]+'gen_rand'))
        for i_batch, batch in enumerate(dataloader):
            if i_batch==max_plots:
                break  
            
            with torch.no_grad():
                # # img_in
                img_in = batch['img_1']
                t_in = batch['time_1']
                # # img_ref
                img_ref = batch['img_2']
                t_ref = batch['time_2']
                # # generate z
                z = torch.Tensor(np.random.normal(0, 1, (img_in.shape[0],cfg['dim_z'])))

                # # run model
                img_gen = model(img_in=img_in,t_in=t_in,t_ref=t_ref,z=z)
                
                # # stack images
                img_combo = torch.cat((img_in,img_gen.to(img_in.device),img_ref),dim=2)
                # # build grid
                grid = torchvision.utils.make_grid(cfg['deNorm'](img_combo[:n_imgs,:]))
                img = cfg['toPIL'](grid)
                
                # # plot
                fig, axs = plt.subplots()
                axs.imshow(img)
                axs.set_title('in-gen-ref ' + str((t_ref-t_in)[:n_imgs].tolist()))
                axs.set_axis_off()
                plt.savefig(os.path.join(plot_dir,'gen_rand_'+str(i_batch)), dpi=cfg['plot_dpi'], bbox_inches='tight')
                plt.close(fig)
        
        
        #%% generate imgs with fixed img_in and z while iteratively increasing t
        # # # -> check plant consistency over time
        max_plots=10
        if cfg['data_name'] == 'abd' or cfg['data_name'] == 'abdc':
            t_ref = torch.arange(18,42)
            in_dist = torch.arange(21,39)
        elif cfg['data_name'] == 'grf':
            t_ref = torch.arange(0,74)
            in_dist = torch.tensor((1,8,16,22,27,35,44,50,57,65,69,71))
        elif cfg['data_name'] == 'mix':# or cfg['data_name'] == 'mix-wg':
            t_ref = torch.arange(0,122)
            in_dist = torch.tensor((7,28,42,54,64,71,74,82,99,117,119))
        elif cfg['data_name'] == 'mix-wg':
            t_ref = torch.arange(0,122)
            in_dist = torch.tensor((7,28,42,54,64,71,74,82,99,117,119)) # since trained on CKA, visualize in_dist from there
            # in_dist = torch.tensor((3,24,29,37,50,66,78,88,100,111))
        else:
            print('Wrong data_name.')
        n_imgs = t_ref.shape[0]
        
        plot_dir = utils.make_folder(cfg['pred_dir'],(prfx[count]+'gen_ood_fixed_in_z'))
        for i_batch, batch in enumerate(dataloader):
            if i_batch==max_plots:
                break  
            # # generate z
            z = torch.Tensor(np.random.normal(0, 1, (1,cfg['dim_z'])))
            z = z.repeat(n_imgs,1)
            
            with torch.no_grad():
                # # img_in pick first image of batch and use it for all others
                img_in = batch['img_1'][0, :].unsqueeze(dim=0).repeat(n_imgs,1,1,1)
                t_in = batch['time_1'][0].repeat(n_imgs)
                img_pred = model(img_in=img_in,t_in=t_in,t_ref=t_ref,z=z)
            
            # # colored frame according to in_img, in_dist or ood
            fw = 5 # frame width
            
            # # color in_dist grey
            ind_idx = torch.nonzero(torch.isin(t_ref, in_dist))   
            img_pred[ind_idx,0,0:fw,:]=0
            img_pred[ind_idx,0,256-fw:256,:]=0
            img_pred[ind_idx,0,:,0:fw]=0
            img_pred[ind_idx,0,:,256-fw:256]=0
            img_pred[ind_idx,1,0:fw,:]=0
            img_pred[ind_idx,1,256-fw:256,:]=0
            img_pred[ind_idx,1,:,0:fw]=0
            img_pred[ind_idx,1,:,256-fw:256]=0
            img_pred[ind_idx,2,0:fw,:]=1
            img_pred[ind_idx,2,256-fw:256,:]=1
            img_pred[ind_idx,2,:,0:fw]=1
            img_pred[ind_idx,2,:,256-fw:256]=1
            
            # # color ood red
            ood_idx = torch.nonzero(~torch.isin(t_ref, in_dist))
            img_pred[ood_idx,0,0:fw,:]=1
            img_pred[ood_idx,0,256-fw:256,:]=1
            img_pred[ood_idx,0,:,0:fw]=1
            img_pred[ood_idx,0,:,256-fw:256]=1
            img_pred[ood_idx,1,0:fw,:]=0.3
            img_pred[ood_idx,1,256-fw:256,:]=0.3
            img_pred[ood_idx,1,:,0:fw]=0.3
            img_pred[ood_idx,1,:,256-fw:256]=0.3
            img_pred[ood_idx,2,0:fw,:]=0
            img_pred[ood_idx,2,256-fw:256,:]=0
            img_pred[ood_idx,2,:,0:fw]=0
            img_pred[ood_idx,2,:,256-fw:256]=0
            
            # # color in_img cyan
            in_idx = (t_ref == t_in[0]).nonzero().item()         
            img_pred[in_idx,0,0:fw,:]=0
            img_pred[in_idx,0,256-fw:256,:]=0
            img_pred[in_idx,0,:,0:fw]=0
            img_pred[in_idx,0,:,256-fw:256]=0
            img_pred[in_idx,1,0:fw,:]=1
            img_pred[in_idx,1,256-fw:256,:]=1
            img_pred[in_idx,1,:,0:fw]=1
            img_pred[in_idx,1,:,256-fw:256]=1
            img_pred[in_idx,2,0:fw,:]=1
            img_pred[in_idx,2,256-fw:256,:]=1
            img_pred[in_idx,2,:,0:fw]=1
            img_pred[in_idx,2,:,256-fw:256]=1
            
            
            grid = torchvision.utils.make_grid(cfg['deNorm'](img_pred), nrow=10)
            img = cfg['toPIL'](grid)
            
            fig, axs = plt.subplots()
            axs.imshow(img)
            axs.set_axis_off()
            plt.savefig(os.path.join(plot_dir,'gen_ood_fixed_in_z_'+str(i_batch)), dpi=cfg['plot_dpi']+300, bbox_inches='tight')
            plt.close(fig)


    #%% plot eval ref imgs and DAS
    plot_dir = utils.make_folder(cfg['pred_dir'],('eval_ref'))
    batch = eval_dataset[0]
    n_imgs = batch['img'].shape[0]

    t_ref = torch.tensor(batch['time'])
    img_ref = batch['img'].detach().clone()
        
    fig, axs = plt.subplots(1, n_imgs)
    plt.subplots_adjust(wspace=0.05) # Adjust this value to control the spacing between images
    title_pad = 0.2  # Adjust this value to control the spacing between title and image

    for idx in range(n_imgs):
        axs[idx].imshow(cfg['toPIL'](cfg['deNorm'](img_ref[idx,:])))
        axs[idx].set_title(str(batch['time'][idx]), pad=title_pad)
        axs[idx].set_yticklabels([])
        axs[idx].set_xticklabels([])
        axs[idx].axis('off')
        
    plt.savefig(os.path.join(plot_dir,'eval_timeseries.png'), dpi=cfg['plot_dpi']+100, bbox_inches='tight')
    plt.savefig(os.path.join(plot_dir,'eval_timeseries.pdf'), dpi=cfg['plot_dpi']+100, bbox_inches='tight')
    plt.close(fig)
    

    #%% eval qual plots dataset
    plot_dir = utils.make_folder(cfg['pred_dir'],('eval_gen_fixed_in_z'))
    batch = eval_dataset[0]
    n_imgs = batch['img'].shape[0]
    n_runs = 10 # for z

    for in_idx in range(0,n_imgs):
        img_in = batch['img'][in_idx,:].unsqueeze(dim=0).repeat(n_imgs,1,1,1)
        t_in = torch.tensor(batch['time'])[in_idx].repeat(n_imgs)
        t_ref = torch.tensor(batch['time'])
        img_ref = batch['img'].detach().clone()
        img_pred = torch.empty((n_runs,n_imgs,3,cfg['img_size'],cfg['img_size']))

        for j in range(0,n_runs):
            # # generate z
            z = torch.Tensor(np.random.normal(0, 1, (1,cfg['dim_z'])))
            z = z.repeat(n_imgs,1)
            
            with torch.no_grad():
                img_pred[j,:] = model(img_in=img_in,t_in=t_in,t_ref=t_ref,z=z)
        
        # # use img of first run [0] for visualisation and make them to grid
        grid = torchvision.utils.make_grid(cfg['deNorm'](img_pred[0,:]), nrow=n_imgs)
        img_pred_grid = cfg['toPIL'](grid)
        
        # # compute std over all runs and then mean over all channels and make them to grid
        img_pred_std = torch.mean(torch.std(img_pred, axis=0), axis=1)
        img_pred_std = img_pred_std/torch.max(img_pred_std) # scale std imgs to range [0 1]
        grid = torchvision.utils.make_grid(torch.unsqueeze(img_pred_std,dim=1), nrow=n_imgs)
        img_std_grid = cfg['toPIL'](grid[0,:]) # grid always creates a 3-channel map, but all 3 channels are the same, as input is BW-img -> just use first channel grid[0,:] -> so that we can apply cmap in the end

        # # metrics only for the first generated img AND NOT for all runs! (before manipulating refs with colored boxes :-D)
        l1 = torch.mean(loss_l1(img_pred[0,:], img_ref), [1,2,3], True).cpu()
        ssim = loss_ssim(img_pred[0,:], img_ref).cpu()
        eval_model.to('cpu')
        if cfg['data_name'] == 'abdc':
            pla_diff = evaluate_pla.loss_pla(eval_model(img_pred[0,:]), eval_model(img_ref), threshold=cfg['pla_threshold'], center_threshold=cfg['pla_center_threshold'], normalize=False)*((cfg['GSD_cm']*10)**2)
        elif cfg['data_name'] == 'grf':
            pla_diff = evaluate_pla.loss_pla(eval_model(img_pred[0,:]), eval_model(img_ref), threshold=cfg['pla_threshold'], center_threshold=cfg['pla_center_threshold'], normalize=False)*(cfg['GSD_cm']**2)
        elif cfg['data_name'] == 'mix' or cfg['data_name'] == 'mix-wg':
            biomass_diff = (eval_model(img_pred[0,:])-eval_model(img_ref)).detach()*cfg['evalM_target_transform']
        lpips = []
        for k in range(img_ref.shape[0]):
            lpips.append(loss_lpips(img_pred[0,k,:].unsqueeze(dim=0), img_ref[k,:].unsqueeze(dim=0)).cpu().detach().item())
            
            
        # # print cyan box around in_img of img_ref and make them to grid
        img_ref[in_idx,0,0:10,:]=0
        img_ref[in_idx,0,246:256,:]=0
        img_ref[in_idx,0,:,0:10]=0
        img_ref[in_idx,0,:,246:256]=0
        img_ref[in_idx,1,0:10,:]=1
        img_ref[in_idx,1,246:256,:]=1
        img_ref[in_idx,1,:,0:10]=1
        img_ref[in_idx,1,:,246:256]=1
        img_ref[in_idx,2,0:10,:]=1
        img_ref[in_idx,2,246:256,:]=1
        img_ref[in_idx,2,:,0:10]=1
        img_ref[in_idx,2,:,246:256]=1
        grid = torchvision.utils.make_grid(cfg['deNorm'](img_ref), nrow=n_imgs)
        img_ref_grid = cfg['toPIL'](grid)
        
        
        # # plot everything in one figure
        fig, axs = plt.subplots(5,1)#, figsize=(cfg['figure_width'], cfg['figure_height']), dpi=cfg['plot_dpi'])
        axis_thickness = 0.6
        
        axs[0].imshow(img_ref_grid)
        # axs[0].set_axis_off()
        axs[0].set_yticklabels([])
        axs[0].set_xticklabels([])
        axs[0].set_ylabel("ref")
        axs[0].spines['right'].set_linewidth(axis_thickness)
        axs[0].spines['left'].set_linewidth(axis_thickness)
        axs[0].spines['top'].set_linewidth(axis_thickness)
        axs[0].spines['bottom'].set_linewidth(axis_thickness)
        axs[0].tick_params(axis='both', which='both', width=axis_thickness)
        # axs[0].axis('off')
        
        axs[1].imshow(img_pred_grid)
        # axs[1].set_axis_off()
        axs[1].set_yticklabels([])
        axs[1].set_xticklabels([])
        axs[1].set_ylabel("gen")
        axs[1].spines['right'].set_linewidth(axis_thickness)
        axs[1].spines['left'].set_linewidth(axis_thickness)
        axs[1].spines['top'].set_linewidth(axis_thickness)
        axs[1].spines['bottom'].set_linewidth(axis_thickness)
        axs[1].tick_params(axis='both', which='both', width=axis_thickness)
        # axs[1].axis('off')
        
        axs[2].imshow(img_std_grid,cmap=plt.cm.Blues,vmin=0,vmax=255/4)
        # axs[2].set_axis_off()
        axs[2].set_yticklabels([])
        axs[2].set_xticklabels([])
        axs[2].set_ylabel("std")
        axs[2].spines['right'].set_linewidth(axis_thickness)
        axs[2].spines['left'].set_linewidth(axis_thickness)
        axs[2].spines['top'].set_linewidth(axis_thickness)
        axs[2].spines['bottom'].set_linewidth(axis_thickness)
        axs[2].tick_params(axis='both', which='both', width=axis_thickness)
        # axs[2].axis('off')
        
        new_x = [x + 0.5 for x in range(len(t_ref))]
        
        axs[3].plot(new_x,lpips,c='#ff336b',label='LPIPS')
        axs[3].plot(new_x,ssim,c='#336bff',label='MS-SSIM')
        axs[3].legend(prop={'size': 5}, loc='lower right')
        axs[3].set_ylim(0, 1)
        axs[3].set_xlim(0, len(t_ref))
        axs[3].set_xticklabels([])
        axs[3].spines['right'].set_linewidth(axis_thickness)
        axs[3].spines['left'].set_linewidth(axis_thickness)
        axs[3].spines['top'].set_linewidth(axis_thickness)
        axs[3].spines['bottom'].set_linewidth(axis_thickness)
        axs[3].tick_params(axis='both', which='both', width=axis_thickness)
        axs[3].grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
        
        custom_xticks = np.array(t_ref, dtype=str)
        
        if cfg['data_name'] == 'abdc':
            axs[4].plot(new_x,pla_diff,c='#499f2d',label='$\Delta$PLA (gen-ref) [mm²]')
        elif cfg['data_name'] == 'grf':
            axs[4].plot(new_x,pla_diff,c='#499f2d',label='$\Delta$PLA (gen-ref) [cm²]')
        elif cfg['data_name'] == 'mix' or cfg['data_name'] == 'mix-wg':
            axs[4].plot(new_x,biomass_diff[:,1],c='#95A626',label='$\Delta${BM}$_{\mathrm{SW}}$ (gen-ref) [t/ha]')
            axs[4].plot(new_x,biomass_diff[:,0],c='#26A695',label='$\Delta${BM}$_{\mathrm{FB}}$ (gen-ref) [t/ha]')

        axs[4].legend(prop={'size': 5}, loc='lower right')
        axs[4].set_xlim(0, len(t_ref))
        if cfg['data_name'] == 'grf':
            axs[4].set_xlabel("days after planting (DAP)") # days after planting
        else:
            axs[4].set_xlabel("days after sowing (DAS)") # days after seeding/sowing
        axs[4].set_xticks(new_x)
        axs[4].set_xticklabels(custom_xticks)
        axs[4].spines['right'].set_linewidth(axis_thickness)
        axs[4].spines['left'].set_linewidth(axis_thickness)
        axs[4].spines['top'].set_linewidth(axis_thickness)
        axs[4].spines['bottom'].set_linewidth(axis_thickness)
        axs[4].tick_params(axis='both', which='both', width=axis_thickness)
        axs[4].grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

        # plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.6)
        if cfg['data_name'] == 'abdc':
            plt.subplots_adjust(bottom=0.1, top=0.6)
        elif cfg['data_name'] == 'mix' or cfg['data_name'] == 'mix-wg' or cfg['data_name'] == 'grf':
            plt.subplots_adjust(bottom=0.1, top=0.8) #0.75

        # plt.subplots_adjust(wspace=-0.5, hspace=-0.5)
        plt.savefig(os.path.join(plot_dir,'gen_fixed_in_z_'+str(t_ref[in_idx].item())), dpi=cfg['plot_dpi']+200, bbox_inches='tight')
        plt.close(fig)
        