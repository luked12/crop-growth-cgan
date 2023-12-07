"""
===============================================================================
Get predictions and plots for models coming from train_img_t_cls.py
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
import pandas as pd
from PIL import Image
from utils import utils
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance

from configs.config_test_img_t_cls import cfg

from datasets.mc_datamodule import MixCropDataModule, MixCrop2ImagesDataModule, MixCrop2Images2DatesDataModule
from models.wgangp_img_t_cls_plm import WGANGPModel_img_t_cls

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
    
    #%% dataModule all times
    dataModule = MixCrop2ImagesDataModule(cfg['img_dir'], cfg['info_tab_path'], cfg['wheat_target_path'], cfg['bean_target_path'], cfg['mix_target_path'], cfg['wheat_treatment_path'], cfg['bean_treatment_path'], cfg['mix_treatment_path'], cfg['data_name'], cfg['data_time'], cfg['batch_size'], cfg['n_workers'], cfg['transform_train'], cfg['transform_test'], target_type=cfg['target_type'], target_transform=cfg['target_transform'], in_memory=cfg['in_memory'], val_test_shuffle=cfg['val_test_shuffle'])
    dataModule_2Dates = MixCrop2Images2DatesDataModule(cfg['img_dir'], cfg['info_tab_path'], cfg['wheat_target_path'], cfg['bean_target_path'], cfg['mix_target_path'], cfg['wheat_treatment_path'], cfg['bean_treatment_path'], cfg['mix_treatment_path'], cfg['data_name'], cfg['data_time'], cfg['date_in'], cfg['date_out'], cfg['batch_size'], cfg['n_workers'], cfg['transform_train'], cfg['transform_test'], target_type=cfg['target_type'], target_transform=cfg['target_transform'], in_memory=cfg['in_memory'], val_test_shuffle=cfg['val_test_shuffle'])

    # setup dataModule
    dataModule.prepare_data()
    dataModule.setup()
    dataModule_2Dates.prepare_data()
    dataModule_2Dates.setup()
    
    # show dim and len of different data subsets
    print('---Some Training Stats---')
    print('Input dims:', dataModule.data_dims)
    print('#Traindata:', len(dataModule.train_dataloader().dataset))
    print('#Valdata:', len(dataModule.val_dataloader().dataset))
    print('#Testdata:', len(dataModule.test_dataloader().dataset))    
    
    if cfg['train_results']:
        dataloader_list = [dataModule.test_dataloader(), dataModule.train_dataloader()]
        dataloader_2Dates_list = [dataModule_2Dates.test_dataloader(), dataModule_2Dates.train_dataloader()]
        prfx=['test_','train_']
    else:
        dataloader_list = [dataModule.test_dataloader()]
        dataloader_2Dates_list = [dataModule_2Dates.test_dataloader()]
        prfx=['test_']

    
    #%% load model from checkpoint        
    if cfg['use_model'] == 'wgangp':
        model = WGANGPModel_img_t_cls.load_from_checkpoint(cfg['ckpt_path_pred'])
        
    # # set to eval mode
    model.eval()
    # # sent model to device
    model.to(cfg['device'])
    
    # # Losses
    loss_l1 = torch.nn.L1Loss(reduction='none')
    loss_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0, reduction=None)
    loss_lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True)


    #%% load eval model (Mask R-CNN or Regression model)
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
        biomass_diff = []
        biomass_ref = []
        biomass_gen = []
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
                # # img_in, t_in, if_in, cls_in
                img_in = batch['img_1']
                t_in = batch['time_1']
                cls_in = batch['label']
                
                # # img_ref, t_ref, if_ref, cls_ref
                img_ref = batch['img_2'].to(cfg['device'])
                t_ref = batch['time_2']
                cls_ref = batch['label']
                
                # # save times to list
                t_gen.append(t_ref)
                t_diff.append(t_ref-t_in)
    
                # # run model
                img_gen = model(img_in=img_in,t_in=t_in,t_ref=t_ref,cls_in=cls_in,cls_ref=cls_ref)
                
                # # calculate FID
                fid.update((cfg['deNorm'](img_ref)*255).to(torch.uint8), real=True)
                fid.update((cfg['deNorm'](img_gen)*255).to(torch.uint8), real=False)
                
                # # L1, SSIM, LPIPS, PSNR and EvalM-metric
                l1.append(torch.mean(loss_l1(img_gen, img_ref), [1,2,3]).cpu())
                ssim.append(loss_ssim(img_gen, img_ref).cpu())
                biomass_diff.append(((eval_model(img_gen)-eval_model(img_ref))).cpu().detach()*cfg['evalM_target_transform'])
                biomass_gen.append((eval_model(img_gen)).cpu().detach()*cfg['evalM_target_transform'])
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

        biomass_diff = np.array(torch.stack(biomass_diff).view(-1,2))
        biomass_ref = np.array(torch.stack(biomass_ref).view(-1,2))
        biomass_gen = np.array(torch.stack(biomass_gen).view(-1,2))
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
            
    
    
    #%% start predicting / scoring / plotting for 2Dates 
    for count, dataloader in enumerate(dataloader_2Dates_list):
        
        plot_dir = utils.make_folder(cfg['pred_dir'],(prfx[count]+'simulation_imgs'))
        
        # # visualize different changed treatments
        max_batches = 5
        for i_batch, batch in enumerate(dataloader):
            if i_batch==max_batches:
                break  
    
            with torch.no_grad():
                # # img_in, t_in, if_in, cls_in
                img_in = batch['img_1']
                t_in = batch['time_1']
                cls_in = batch['label']
                
                # # img_ref, t_ref, if_ref, cls_ref
                img_ref = batch['img_2'].to(cfg['device'])
                t_ref = batch['time_2']
                cls_ref = batch['label']
                cls_modify1 = cls_ref+24 # + 24 means change from faba bean type A to faba bean type B
                cls_modify1[cls_modify1<0]=0
                cls_modify1[cls_modify1>75]=75
                
                cls_modify2 = cls_ref-1 # -1 means change low density L to high density H
                cls_modify2[cls_modify2<0]=0
                cls_modify2[cls_modify2>75]=75
                
                cls_modify3 = torch.ones(cfg['batch_size'],dtype=torch.int64) # =1 means change to FB monoculture type A
                
                cls_modify4 = cls_ref-24 # -24 means change to SW monoculture with cultivar which was included in the mixture
                cls_modify4[cls_modify4<0]=0
                cls_modify4[cls_modify4>75]=75
    
                # # run model
                z = torch.Tensor(np.random.normal(0, 1, (cfg['batch_size'],cfg['dim_z'])))
                img_gen_ref = model(img_in=img_in,t_in=t_in,t_ref=t_ref,cls_in=cls_in,cls_ref=cls_ref,z=z) # # original
                img_gen_1 = model(img_in=img_in,t_in=t_in,t_ref=t_ref,cls_in=cls_in,cls_ref=cls_modify1,z=z) # # change
                img_gen_2 = model(img_in=img_in,t_in=t_in,t_ref=t_ref,cls_in=cls_in,cls_ref=cls_modify2,z=z) # # change
                img_gen_3 = model(img_in=img_in,t_in=t_in,t_ref=t_ref,cls_in=cls_in,cls_ref=cls_modify3,z=z) # # change
                img_gen_4 = model(img_in=img_in,t_in=t_in,t_ref=t_ref,cls_in=cls_in,cls_ref=cls_modify4,z=z) # # change
                
                # # # plot
                for j in range(cfg['batch_size']):
                    # plot only mixture where the input was faba bean (A) and low density (L) treatment
                    if cls_in[j]>27 and cls_in[j]<52 and cls_in[j] % 2 != 0:
                        cfg['toPIL'](cfg['deNorm'](img_in[j,:])).save(os.path.join(plot_dir,str(i_batch)+'_'+str(j)+'_'+str(cls_in[j].item())+'_in.png'))
                        cfg['toPIL'](cfg['deNorm'](img_ref[j,:])).save(os.path.join(plot_dir,str(i_batch)+'_'+str(j)+'_'+str(cls_in[j].item())+'_ref.png'))
                        cfg['toPIL'](cfg['deNorm'](img_gen_ref[j,:])).save(os.path.join(plot_dir,str(i_batch)+'_'+str(j)+'_'+str(cls_in[j].item())+'_genref.png'))
                        cfg['toPIL'](cfg['deNorm'](img_gen_1[j,:])).save(os.path.join(plot_dir,str(i_batch)+'_'+str(j)+'_'+str(cls_in[j].item())+'_gen1.png'))
                        cfg['toPIL'](cfg['deNorm'](img_gen_2[j,:])).save(os.path.join(plot_dir,str(i_batch)+'_'+str(j)+'_'+str(cls_in[j].item())+'_gen2.png'))
                        cfg['toPIL'](cfg['deNorm'](img_gen_3[j,:])).save(os.path.join(plot_dir,str(i_batch)+'_'+str(j)+'_'+str(cls_in[j].item())+'_gen3.png'))
                        cfg['toPIL'](cfg['deNorm'](img_gen_4[j,:])).save(os.path.join(plot_dir,str(i_batch)+'_'+str(j)+'_'+str(cls_in[j].item())+'_gen4.png'))


        #%% plot dir for all simulations
        plot_dir = utils.make_folder(cfg['pred_dir'],(prfx[count]+'simulations'))
        
        #%% calculate metrics with changed faba bean type A->B
        t_gen = []
        t_diff = []
        biomass_gen_0 = []
        biomass_gen_1 = []
        label = []
        
        # # avoid going into the last batch as it is sometimes < batch_size and this causes problems later on
        max_batches = math.floor(len(dataloader.dataset)/cfg['batch_size'])
        # max_batches = 2
        for i_batch, batch in enumerate(dataloader):
            if i_batch==max_batches:
                break  
            
            with torch.no_grad():
                # # img_in, t_in, if_in, cls_in
                img_in = batch['img_1']
                t_in = batch['time_1']
                cls_in = batch['label']
                
                # # img_ref, t_ref, if_ref, cls_ref
                img_ref = batch['img_2'].to(cfg['device'])
                t_ref = batch['time_2']
                cls_ref = batch['label']
                cls_modify = cls_ref+24 # + 24 means change from faba bean type A to faba bean type B
                cls_modify[cls_modify<0]=0
                cls_modify[cls_modify>75]=75
                # # save times to list
                t_gen.append(t_ref)
                t_diff.append(t_ref-t_in)
    
                # # run model
                z = torch.Tensor(np.random.normal(0, 1, (cfg['batch_size'],cfg['dim_z'])))
                img_gen_0 = model(img_in=img_in,t_in=t_in,t_ref=t_ref,cls_in=cls_in,cls_ref=cls_ref,z=z) # # original
                img_gen_1 = model(img_in=img_in,t_in=t_in,t_ref=t_ref,cls_in=cls_in,cls_ref=cls_modify,z=z) # # change 
                
                # # # plot
                # cfg['toPIL'](cfg['deNorm'](img_gen_0[1,:])).save(os.path.join(plot_dir,'gen.png'))
                
                # # EvalM-metric
                biomass_gen_0.append((eval_model(img_gen_0)).cpu().detach()*cfg['evalM_target_transform'])
                biomass_gen_1.append((eval_model(img_gen_1)).cpu().detach()*cfg['evalM_target_transform'])
                label.append(cls_in)
                
        t_gen = np.array(torch.cat(t_gen))
        t_diff = np.array(torch.cat(t_diff))

        biomass_gen_0 = np.array(torch.stack(biomass_gen_0).view(-1,2))
        biomass_gen_1 = np.array(torch.stack(biomass_gen_1).view(-1,2))
        label = np.array(torch.cat(label))
            
        modify_labels = range(28,52)    
        # modify_labels = [1,2,3,4] # # could be also manual given
        
        # # stuff to load simulated biomass for orig and modified treatment
        mix_target_df = pd.read_csv(cfg['mix_target_path'])
        mix_treatment_df = pd.read_csv(cfg['mix_treatment_path'])
    
        orig_trt = ['Mix_A_1_H', 'Mix_A_1_L','Mix_A_2_H', 'Mix_A_2_L','Mix_A_3_H', 'Mix_A_3_L','Mix_A_4_H', 'Mix_A_4_L','Mix_A_5_H', 'Mix_A_5_L','Mix_A_6_H', 'Mix_A_6_L','Mix_A_7_H', 'Mix_A_7_L','Mix_A_8_H', 'Mix_A_8_L','Mix_A_9_H', 'Mix_A_9_L','Mix_A_10_H', 'Mix_A_10_L','Mix_A_11_H', 'Mix_A_11_L','Mix_A_12_H', 'Mix_A_12_L',]
        modify_trt = ['Mix_B_1_H', 'Mix_B_1_L','Mix_B_2_H', 'Mix_B_2_L','Mix_B_3_H', 'Mix_B_3_L','Mix_B_4_H', 'Mix_B_4_L','Mix_B_5_H', 'Mix_B_5_L','Mix_B_6_H', 'Mix_B_6_L','Mix_B_7_H', 'Mix_B_7_L','Mix_B_8_H', 'Mix_B_8_L','Mix_B_9_H', 'Mix_B_9_L','Mix_B_10_H', 'Mix_B_10_L','Mix_B_11_H', 'Mix_B_11_L','Mix_B_12_H', 'Mix_B_12_L',]        
        
        date_ = cfg['date_out'][-2:]+'.'+cfg['date_out'][-5:-3]+'.'+cfg['date_out'][0:4]
        scores = {}
        for idx, modify_label in enumerate(modify_labels):
            scores[idx] = {}
            # estimated biomass for orig trt
            scores[idx]['bm_gen_0'] = np.mean(biomass_gen_0[label==modify_label],axis=0)
            scores[idx]['bm_gen_0_std'] = np.std(biomass_gen_0[label==modify_label],axis=0)
            # estimated biomass for modify trt
            scores[idx]['bm_gen_1'] = np.mean(biomass_gen_1[label==modify_label],axis=0)
            scores[idx]['bm_gen_1_std'] = np.std(biomass_gen_1[label==modify_label],axis=0)
            
            # simulated biomass for orig trt
            simulation_id = mix_treatment_df.loc[mix_treatment_df['orig_trt'] == orig_trt[idx]].loc[mix_treatment_df['Location'] == cfg['site']]['projectid'].values[0]
            biomass_W = mix_target_df.loc[mix_target_df['projectid'] == simulation_id].loc[mix_target_df['CURRENT.DATE'] == date_]['AGBG_1_t_ha'].values[0]
            biomass_B = mix_target_df.loc[mix_target_df['projectid'] == simulation_id].loc[mix_target_df['CURRENT.DATE'] == date_]['AGBG_2_t_ha'].values[0]
            scores[idx]['bm_simu_0'] = np.array((biomass_B,biomass_W))
            
            # simulated biomass for modify trt
            simulation_id = mix_treatment_df.loc[mix_treatment_df['orig_trt'] == modify_trt[idx]].loc[mix_treatment_df['Location'] == cfg['site']]['projectid'].values[0]
            biomass_W = mix_target_df.loc[mix_target_df['projectid'] == simulation_id].loc[mix_target_df['CURRENT.DATE'] == date_]['AGBG_1_t_ha'].values[0]
            biomass_B = mix_target_df.loc[mix_target_df['projectid'] == simulation_id].loc[mix_target_df['CURRENT.DATE'] == date_]['AGBG_2_t_ha'].values[0]
            scores[idx]['bm_simu_1'] = np.array((biomass_B,biomass_W))


        # # re-group for plotting
        bm_gen_fb = [[] for _ in range(2)]
        bm_gen_fb_std = [[] for _ in range(2)]
        bm_gen_sw = [[] for _ in range(2)]
        bm_gen_sw_std = [[] for _ in range(2)]
        bm_simu_fb = [[] for _ in range(2)]
        bm_simu_sw = [[] for _ in range(2)]
        for i in range(0,24):
            bm_gen_fb[0].append(scores[i]['bm_gen_0'][0])
            bm_gen_fb[1].append(scores[i]['bm_gen_1'][0])
            bm_gen_fb_std[0].append(scores[i]['bm_gen_0_std'][0])
            bm_gen_fb_std[1].append(scores[i]['bm_gen_1_std'][0])
        
            bm_gen_sw[0].append(scores[i]['bm_gen_0'][1])
            bm_gen_sw[1].append(scores[i]['bm_gen_1'][1])
            bm_gen_sw_std[0].append(scores[i]['bm_gen_0_std'][1])
            bm_gen_sw_std[1].append(scores[i]['bm_gen_1_std'][1])
            
            bm_simu_fb[0].append(scores[i]['bm_simu_0'][0])
            bm_simu_fb[1].append(scores[i]['bm_simu_1'][0])
            
            bm_simu_sw[0].append(scores[i]['bm_simu_0'][1])
            bm_simu_sw[1].append(scores[i]['bm_simu_1'][1])
            
            
        # Sample data
        x_tick_labels = orig_trt
        x = np.arange(len(x_tick_labels))  # x-coordinates for sets
        width = 0.4  # Width of the bars
        
        # Create the grouped stacked bar plot
        fig, ax = plt.subplots(figsize=(cfg['figure_width']*2, cfg['figure_height']), dpi=cfg['plot_dpi'])
        
        # Stack the first set of estimated values
        for i, (est_values, std_values) in enumerate(zip(bm_gen_fb, bm_gen_fb_std)):
            if i==0:
                ax.bar(x + (i - 0.5) * width, est_values, yerr=std_values, width=width, color='#26A695', edgecolor='#26A695', label='{BM}$_{\mathrm{FB}}$ A to A')
            elif i==1:
                ax.bar(x + (i - 0.5) * width, est_values, yerr=std_values, width=width, color='None', edgecolor='#26A695', hatch='//', label='{BM}$_{\mathrm{FB}}$ A to B')
        
        # Stack the second set of estimated values on top of the first set
        for i, (est_values, std_values) in enumerate(zip(bm_gen_sw, bm_gen_sw_std)):
            if i==0:
                ax.bar(x + (i - 0.5) * width, est_values, yerr=std_values, width=width, bottom=bm_gen_fb[i], color='#95A626', edgecolor='#95A626', label='{BM}$_{\mathrm{SW}}$ A to A')
            elif i==1:
                ax.bar(x + (i - 0.5) * width, est_values, yerr=std_values, width=width, bottom=bm_gen_fb[i], color='None', edgecolor='#95A626', hatch='//', label='{BM}$_{\mathrm{SW}}$ A to B')
        
        # Add reference points
        for i, (ref_values_1, ref_values_2) in enumerate(zip(bm_simu_fb, bm_simu_sw)):
            if i==0:
                ax.scatter(x + (i - 0.5) * width, ref_values_1, marker='.', color='#ff336b', label='{BM}$_{\mathrm{FB}}$ Ref')
                ax.scatter(x + (i - 0.5) * width, np.array(ref_values_2)+np.array(bm_gen_fb[i]), marker='x', color='#ff336b', label='{BM}$_{\mathrm{SW}}$ Ref')
            else:
                ax.scatter(x + (i - 0.5) * width, ref_values_1, marker='.', color='#ff336b')
                ax.scatter(x + (i - 0.5) * width, np.array(ref_values_2)+np.array(bm_gen_fb[i]), marker='x', color='#ff336b')

        ax.set_xlabel('Original input image treatment')        
        ax.set_ylabel('Biomass [t/ha]')
        ax.set_title('')
        ax.set_xticks(x)
        ax.set_xticklabels(x_tick_labels, rotation=90)
        # ax.set_ylim((0,2))
        ax.legend(loc='upper center', ncol=3)
        plt.savefig(os.path.join(plot_dir,'AB_change_stacked'), dpi=cfg['plot_dpi']+200, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        
        
        # Create the grouped FB bar plot
        fig, ax = plt.subplots(figsize=(cfg['figure_width']*2, cfg['figure_height']), dpi=cfg['plot_dpi'])
        
        # Stack the first set of estimated values
        for i, (est_values, std_values) in enumerate(zip(bm_gen_fb, bm_gen_fb_std)):
            if i==0:
                ax.bar(x + (i - 0.5) * width, est_values, yerr=std_values, width=width, color='#26A695', edgecolor='#26A695', label='{BM}$_{\mathrm{FB}}$ A to A')
            elif i==1:
                ax.bar(x + (i - 0.5) * width, est_values, yerr=std_values, width=width, color='None', edgecolor='#26A695', hatch='//', label='{BM}$_{\mathrm{FB}}$ A to B')
        
        # Add reference points
        for i, ref_values_1 in enumerate(bm_simu_fb):
            if i==0:
                ax.scatter(x + (i - 0.5) * width, ref_values_1, marker='.', color='#ff336b', label='{BM}$_{\mathrm{FB}}$ Ref')
            else:
                ax.scatter(x + (i - 0.5) * width, ref_values_1, marker='.', color='#ff336b')

        ax.set_xlabel('Original input image treatment')        
        ax.set_ylabel('Biomass [t/ha]')
        ax.set_title('')
        ax.set_xticks(x)
        ax.set_xticklabels(x_tick_labels, rotation=90)
        # ax.set_ylim((0,1.6))
        ax.set_ylim((0,8))
        ax.legend(loc='upper center', ncol=3)
        plt.savefig(os.path.join(plot_dir,'AB_change_FB'), dpi=cfg['plot_dpi']+200, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        
        
        # Create the grouped SW bar plot
        fig, ax = plt.subplots(figsize=(cfg['figure_width']*2, cfg['figure_height']), dpi=cfg['plot_dpi'])
        
        # Stack the second set of estimated values on top of the first set
        for i, (est_values, std_values) in enumerate(zip(bm_gen_sw, bm_gen_sw_std)):
            if i==0:
                ax.bar(x + (i - 0.5) * width, est_values, yerr=std_values, width=width, color='#95A626', edgecolor='#95A626', label='{BM}$_{\mathrm{SW}}$ A to A')
            elif i==1:
                ax.bar(x + (i - 0.5) * width, est_values, yerr=std_values, width=width, color='None', edgecolor='#95A626', hatch='//', label='{BM}$_{\mathrm{SW}}$ A to B')

        # Add reference points
        for i, ref_values_1 in enumerate(bm_simu_sw):
            if i==0:
                ax.scatter(x + (i - 0.5) * width, ref_values_1, marker='.', color='#ff336b', label='{BM}$_{\mathrm{SW}}$ Ref')
            else:
                ax.scatter(x + (i - 0.5) * width, ref_values_1, marker='.', color='#ff336b')
                
        ax.set_xlabel('Original input image treatment')        
        ax.set_ylabel('Biomass [t/ha]')
        ax.set_title('')
        ax.set_xticks(x)
        ax.set_xticklabels(x_tick_labels, rotation=90)
        # ax.set_ylim((0,1.6))
        ax.set_ylim((0,8))
        ax.legend(loc='upper center', ncol=3)
        plt.savefig(os.path.join(plot_dir,'AB_change_SW'), dpi=cfg['plot_dpi']+200, bbox_inches='tight')
        plt.show()
        plt.close(fig)



        #%% calculate metrics with changed density (L->H)
        t_gen = []
        t_diff = []
        biomass_gen_0 = []
        biomass_gen_1 = []
        label = []
        
        # # avoid going into the last batch as it is sometimes < batch_size and this causes problems later on
        max_batches = math.floor(len(dataloader.dataset)/cfg['batch_size'])
        # max_batches = 2
        for i_batch, batch in enumerate(dataloader):
            if i_batch==max_batches:
                break  
            
            with torch.no_grad():
                # # img_in, t_in, if_in, cls_in
                img_in = batch['img_1']
                t_in = batch['time_1']
                cls_in = batch['label']
                
                # # img_ref, t_ref, if_ref, cls_ref
                img_ref = batch['img_2'].to(cfg['device'])
                t_ref = batch['time_2']
                cls_ref = batch['label']
                cls_modify = cls_ref-1 # change from low density to high density
                cls_modify[cls_modify<0]=0
                cls_modify[cls_modify>75]=75
                # # save times to list
                t_gen.append(t_ref)
                t_diff.append(t_ref-t_in)
        
                # # run model
                z = torch.Tensor(np.random.normal(0, 1, (cfg['batch_size'],cfg['dim_z'])))
                img_gen_0 = model(img_in=img_in,t_in=t_in,t_ref=t_ref,cls_in=cls_in,cls_ref=cls_ref,z=z) # # original
                img_gen_1 = model(img_in=img_in,t_in=t_in,t_ref=t_ref,cls_in=cls_in,cls_ref=cls_modify,z=z) # # change
        
                # # EvalM-metric
                biomass_gen_0.append((eval_model(img_gen_0)).cpu().detach()*cfg['evalM_target_transform'])
                biomass_gen_1.append((eval_model(img_gen_1)).cpu().detach()*cfg['evalM_target_transform'])
                label.append(cls_in)
                
        t_gen = np.array(torch.cat(t_gen))
        t_diff = np.array(torch.cat(t_diff))
        
        biomass_gen_0 = np.array(torch.stack(biomass_gen_0).view(-1,2))
        biomass_gen_1 = np.array(torch.stack(biomass_gen_1).view(-1,2))
        label = np.array(torch.cat(label))
            
        modify_labels = [29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,75]    
        
        # # stuff to load simulated biomass for orig and modified treatment
        mix_target_df = pd.read_csv(cfg['mix_target_path'])
        mix_treatment_df = pd.read_csv(cfg['mix_treatment_path'])
        
        orig_trt = ['Mix_A_1_L','Mix_A_2_L','Mix_A_3_L','Mix_A_4_L','Mix_A_5_L','Mix_A_6_L','Mix_A_7_L','Mix_A_8_L','Mix_A_9_L','Mix_A_10_L','Mix_A_11_L','Mix_A_12_L','Mix_B_1_L','Mix_B_2_L','Mix_B_3_L','Mix_B_4_L','Mix_B_5_L','Mix_B_6_L','Mix_B_7_L','Mix_B_8_L','Mix_B_9_L','Mix_B_10_L','Mix_B_11_L','Mix_B_12_L',]
        modify_trt = ['Mix_A_1_H','Mix_A_2_H','Mix_A_3_H','Mix_A_4_H','Mix_A_5_H','Mix_A_6_H','Mix_A_7_H','Mix_A_8_H','Mix_A_9_H','Mix_A_10_H','Mix_A_11_H','Mix_A_12_H','Mix_B_1_H','Mix_B_2_H','Mix_B_3_H','Mix_B_4_H','Mix_B_5_H','Mix_B_6_H','Mix_B_7_H','Mix_B_8_H','Mix_B_9_H','Mix_B_10_H','Mix_B_11_H','Mix_B_12_H',]
        
        date_ = cfg['date_out'][-2:]+'.'+cfg['date_out'][-5:-3]+'.'+cfg['date_out'][0:4]
        scores = {}
        for idx, modify_label in enumerate(modify_labels):
            scores[idx] = {}
            # estimated biomass for orig trt
            scores[idx]['bm_gen_0'] = np.mean(biomass_gen_0[label==modify_label],axis=0)
            scores[idx]['bm_gen_0_std'] = np.std(biomass_gen_0[label==modify_label],axis=0)
            # estimated biomass for modify trt
            scores[idx]['bm_gen_1'] = np.mean(biomass_gen_1[label==modify_label],axis=0)
            scores[idx]['bm_gen_1_std'] = np.std(biomass_gen_1[label==modify_label],axis=0)
            
            # simulated biomass for orig trt
            simulation_id = mix_treatment_df.loc[mix_treatment_df['orig_trt'] == orig_trt[idx]].loc[mix_treatment_df['Location'] == cfg['site']]['projectid'].values[0]
            biomass_W = mix_target_df.loc[mix_target_df['projectid'] == simulation_id].loc[mix_target_df['CURRENT.DATE'] == date_]['AGBG_1_t_ha'].values[0]
            biomass_B = mix_target_df.loc[mix_target_df['projectid'] == simulation_id].loc[mix_target_df['CURRENT.DATE'] == date_]['AGBG_2_t_ha'].values[0]
            scores[idx]['bm_simu_0'] = np.array((biomass_B,biomass_W))
            
            # simulated biomass for modify trt
            simulation_id = mix_treatment_df.loc[mix_treatment_df['orig_trt'] == modify_trt[idx]].loc[mix_treatment_df['Location'] == cfg['site']]['projectid'].values[0]
            biomass_W = mix_target_df.loc[mix_target_df['projectid'] == simulation_id].loc[mix_target_df['CURRENT.DATE'] == date_]['AGBG_1_t_ha'].values[0]
            biomass_B = mix_target_df.loc[mix_target_df['projectid'] == simulation_id].loc[mix_target_df['CURRENT.DATE'] == date_]['AGBG_2_t_ha'].values[0]
            scores[idx]['bm_simu_1'] = np.array((biomass_B,biomass_W))
        
        
        # # re-group for plotting
        bm_gen_fb = [[] for _ in range(2)]
        bm_gen_fb_std = [[] for _ in range(2)]
        bm_gen_sw = [[] for _ in range(2)]
        bm_gen_sw_std = [[] for _ in range(2)]
        bm_simu_fb = [[] for _ in range(2)]
        bm_simu_sw = [[] for _ in range(2)]
        for i in range(0,24):
            bm_gen_fb[0].append(scores[i]['bm_gen_0'][0])
            bm_gen_fb[1].append(scores[i]['bm_gen_1'][0])
            bm_gen_fb_std[0].append(scores[i]['bm_gen_0_std'][0])
            bm_gen_fb_std[1].append(scores[i]['bm_gen_1_std'][0])
        
            bm_gen_sw[0].append(scores[i]['bm_gen_0'][1])
            bm_gen_sw[1].append(scores[i]['bm_gen_1'][1])
            bm_gen_sw_std[0].append(scores[i]['bm_gen_0_std'][1])
            bm_gen_sw_std[1].append(scores[i]['bm_gen_1_std'][1])
            
            bm_simu_fb[0].append(scores[i]['bm_simu_0'][0])
            bm_simu_fb[1].append(scores[i]['bm_simu_1'][0])
            
            bm_simu_sw[0].append(scores[i]['bm_simu_0'][1])
            bm_simu_sw[1].append(scores[i]['bm_simu_1'][1])
            
        
        # Sample data
        x_tick_labels = orig_trt
        x = np.arange(len(x_tick_labels))  # x-coordinates for sets
        width = 0.4  # Width of the bars
        
        # Create the grouped stacked bar plot
        fig, ax = plt.subplots(figsize=(cfg['figure_width']*2, cfg['figure_height']), dpi=cfg['plot_dpi'])
        
        # Stack the first set of estimated values
        for i, (est_values, std_values) in enumerate(zip(bm_gen_fb, bm_gen_fb_std)):
            if i==0:
                ax.bar(x + (i - 0.5) * width, est_values, yerr=std_values, width=width, color='#26A695', edgecolor='#26A695', label='{BM}$_{\mathrm{FB}}$ L to L')
            elif i==1:
                ax.bar(x + (i - 0.5) * width, est_values, yerr=std_values, width=width, color='None', edgecolor='#26A695', hatch='//', label='{BM}$_{\mathrm{FB}}$ L to H')
        
        # Stack the second set of estimated values on top of the first set
        for i, (est_values, std_values) in enumerate(zip(bm_gen_sw, bm_gen_sw_std)):
            if i==0:
                ax.bar(x + (i - 0.5) * width, est_values, yerr=std_values, width=width, bottom=bm_gen_fb[i], color='#95A626', edgecolor='#95A626', label='{BM}$_{\mathrm{SW}}$ L to L')
            elif i==1:
                ax.bar(x + (i - 0.5) * width, est_values, yerr=std_values, width=width, bottom=bm_gen_fb[i], color='None', edgecolor='#95A626', hatch='//', label='{BM}$_{\mathrm{SW}}$ L to H')
        
        # Add reference points
        for i, (ref_values_1, ref_values_2) in enumerate(zip(bm_simu_fb, bm_simu_sw)):
            if i==0:
                ax.scatter(x + (i - 0.5) * width, ref_values_1, marker='.', color='#ff336b', label='{BM}$_{\mathrm{FB}}$ Ref')
                ax.scatter(x + (i - 0.5) * width, np.array(ref_values_2)+np.array(bm_gen_fb[i]), marker='x', color='#ff336b', label='{BM}$_{\mathrm{SW}}$ Ref')
            else:
                ax.scatter(x + (i - 0.5) * width, ref_values_1, marker='.', color='#ff336b')
                ax.scatter(x + (i - 0.5) * width, np.array(ref_values_2)+np.array(bm_gen_fb[i]), marker='x', color='#ff336b')
        
        ax.set_xlabel('Original input image treatment')        
        ax.set_ylabel('Biomass [t/ha]')
        ax.set_title('')
        ax.set_xticks(x)
        ax.set_xticklabels(x_tick_labels, rotation=90)
        # ax.set_ylim((0,2))
        ax.legend(loc='upper center', ncol=3)
        plt.savefig(os.path.join(plot_dir,'LH_change_stacked'), dpi=cfg['plot_dpi']+200, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        
        
        # Create the grouped FB bar plot
        fig, ax = plt.subplots(figsize=(cfg['figure_width']*2, cfg['figure_height']), dpi=cfg['plot_dpi'])
        
        # Stack the first set of estimated values
        for i, (est_values, std_values) in enumerate(zip(bm_gen_fb, bm_gen_fb_std)):
            if i==0:
                ax.bar(x + (i - 0.5) * width, est_values, yerr=std_values, width=width, color='#26A695', edgecolor='#26A695', label='{BM}$_{\mathrm{FB}}$ L to L')
            elif i==1:
                ax.bar(x + (i - 0.5) * width, est_values, yerr=std_values, width=width, color='None', edgecolor='#26A695', hatch='//', label='{BM}$_{\mathrm{FB}}$ L to H')
        
        # Add reference points
        for i, ref_values_1 in enumerate(bm_simu_fb):
            if i==0:
                ax.scatter(x + (i - 0.5) * width, ref_values_1, marker='.', color='#ff336b', label='{BM}$_{\mathrm{FB}}$ Ref')
            else:
                ax.scatter(x + (i - 0.5) * width, ref_values_1, marker='.', color='#ff336b')
        
        ax.set_xlabel('Original input image treatment')        
        ax.set_ylabel('Biomass [t/ha]')
        ax.set_title('')
        ax.set_xticks(x)
        ax.set_xticklabels(x_tick_labels, rotation=90)
        # ax.set_ylim((0,1.6))
        ax.set_ylim((0,8))
        ax.legend(loc='upper center', ncol=3)
        plt.savefig(os.path.join(plot_dir,'LH_change_FB'), dpi=cfg['plot_dpi']+200, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        
        
        # Create the grouped SW bar plot
        fig, ax = plt.subplots(figsize=(cfg['figure_width']*2, cfg['figure_height']), dpi=cfg['plot_dpi'])
        
        # Stack the second set of estimated values on top of the first set
        for i, (est_values, std_values) in enumerate(zip(bm_gen_sw, bm_gen_sw_std)):
            if i==0:
                ax.bar(x + (i - 0.5) * width, est_values, yerr=std_values, width=width, color='#95A626', edgecolor='#95A626', label='{BM}$_{\mathrm{SW}}$ L to L')
            elif i==1:
                ax.bar(x + (i - 0.5) * width, est_values, yerr=std_values, width=width, color='None', edgecolor='#95A626', hatch='//', label='{BM}$_{\mathrm{SW}}$ L to H')
        
        # Add reference points
        for i, ref_values_1 in enumerate(bm_simu_sw):
            if i==0:
                ax.scatter(x + (i - 0.5) * width, ref_values_1, marker='.', color='#ff336b', label='{BM}$_{\mathrm{SW}}$ Ref')
            else:
                ax.scatter(x + (i - 0.5) * width, ref_values_1, marker='.', color='#ff336b')
                
        ax.set_xlabel('Original input image treatment')        
        ax.set_ylabel('Biomass [t/ha]')
        ax.set_title('')
        ax.set_xticks(x)
        ax.set_xticklabels(x_tick_labels, rotation=90)
        # ax.set_ylim((0,1.6))
        ax.set_ylim((0,8))
        ax.legend(loc='upper center', ncol=3)
        plt.savefig(os.path.join(plot_dir,'LH_change_SW'), dpi=cfg['plot_dpi']+200, bbox_inches='tight')
        plt.show()
        plt.close(fig)
