"""
===============================================================================
Get predictions and plots for models coming from train_img_t_if_cls.py
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

from configs.config_test_img_t_if_cls import cfg

from datasets.rgb_plant_image_dataset import RGBPlantSeqDataset
from datasets.mc_datamodule import MixCropDataModule, MixCrop2ImagesDataModule, MixCrop2Images2DatesDataModule
# from models.gan_img_t_if_plm import GANModel_img_t_if
from models.wgangp_img_t_if_cls_plm import WGANGPModel_img_t_if_cls

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
    dataModule = MixCrop2ImagesDataModule(cfg['img_dir'], cfg['info_tab_path'], cfg['wheat_target_path'], cfg['bean_target_path'], cfg['mix_target_path'], cfg['wheat_treatment_path'], cfg['bean_treatment_path'], cfg['mix_treatment_path'], cfg['data_name'], cfg['data_time'], cfg['batch_size'], cfg['n_workers'], cfg['transform_train'], cfg['transform_test'], target_type=cfg['target_type'], target_transform=cfg['target_transform'], in_memory=cfg['in_memory'], val_test_shuffle=cfg['val_test_shuffle'])
    # eval_dataset = RGBPlantSeqDataset(cfg['img_dir']+'eval/', cfg['data_name'], cfg['data_time'], transform=cfg['transform_test'])

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
        model = GANModel_img_t_if_cls.load_from_checkpoint(cfg['ckpt_path_pred'])
    elif cfg['use_model'] == 'wgangp':
        model = WGANGPModel_img_t_if_cls.load_from_checkpoint(cfg['ckpt_path_pred'])
        
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
        
        # # plot dir for scores
        plot_dir = utils.make_folder(cfg['pred_dir'],(prfx[count]+'metrics'))
    
        #%% calculate metrics
        # fid = FrechetInceptionDistance(feature=2048).to(cfg['device'])
        # t_gen = []
        # t_diff = []
        # l1 = []
        # ssim = []
        # lpips = []
        # psnr = []
        # pla_diff = []
        # pla_diff_norm = []
        # pla_ref = []
        # biomass_diff = []
        # biomass_ref = []
        # biomass_gen = []
        # label = []
        
        # # # avoid going into the last batch as it is sometimes < batch_size and this causes problems later on
        # max_batches = math.floor(len(dataloader.dataset)/cfg['batch_size'])
        # # max_batches = 1
        # for i_batch, batch in enumerate(dataloader):
        #     if i_batch==max_batches:
        #         break  
            
        #     with torch.no_grad():
        #         # # img_in, t_in, if_in, cls_in
        #         img_in = batch['img_1']
        #         t_in = batch['time_1']
        #         if_in = batch['biomass_1']
        #         cls_in = batch['label']
                
        #         # # img_ref, t_ref, if_ref, cls_ref
        #         img_ref = batch['img_2'].to(cfg['device'])
        #         t_ref = batch['time_2']
        #         if_ref = batch['biomass_2']
        #         cls_ref = batch['label']
                
        #         # # save times to list
        #         t_gen.append(t_ref)
        #         t_diff.append(t_ref-t_in)
    
        #         # # run model
        #         img_gen = model(img_in=img_in,t_in=t_in,t_ref=t_ref,if_in=if_in,if_ref=if_ref,cls_in=cls_in,cls_ref=cls_ref)
                
        #         # # calculate FID
        #         fid.update((cfg['deNorm'](img_ref)*255).to(torch.uint8), real=True)
        #         fid.update((cfg['deNorm'](img_gen)*255).to(torch.uint8), real=False)
                
        #         # # L1, SSIM, LPIPS, PSNR and EvalM-metric
        #         l1.append(torch.mean(loss_l1(img_gen, img_ref), [1,2,3]).cpu())
        #         ssim.append(loss_ssim(img_gen, img_ref).cpu())
        #         biomass_diff.append(((eval_model(img_gen)-eval_model(img_ref))).cpu().detach()*cfg['evalM_target_transform'])
        #         biomass_gen.append((eval_model(img_gen)).cpu().detach()*cfg['evalM_target_transform'])
        #         biomass_ref.append((eval_model(img_ref)).cpu().detach()*cfg['evalM_target_transform'])
        #         label.append(batch['label'])
        #         for k in range(img_ref.shape[0]):
        #             lpips.append(loss_lpips(img_gen[k,:].unsqueeze(dim=0).cpu().detach(), img_ref[k,:].unsqueeze(dim=0).cpu().detach()).item())
        #             psnr.append(utils.calculate_psnr(img_gen[k,:].cpu().detach(),img_ref[k,:].cpu().detach(), max_value=1).item())
                
        # t_gen = np.array(torch.cat(t_gen))
        # t_diff = np.array(torch.cat(t_diff))
        # t0_filter = np.where((abs(t_diff)==0))[0]
        # t1_filter = np.where((abs(t_diff)>0) & (abs(t_diff)<=10))[0]
        # t2_filter = np.where((abs(t_diff)>10))[0]
        # l1 = np.array(torch.cat(l1))
        # ssim = np.array(torch.cat(ssim))
        # lpips = np.array(lpips)
        # psnr = np.array(psnr)

        # biomass_diff = np.array(torch.stack(biomass_diff).view(-1,2))
        # biomass_ref = np.array(torch.stack(biomass_ref).view(-1,2))
        # biomass_gen = np.array(torch.stack(biomass_gen).view(-1,2))
        # label = np.array(torch.cat(label))
        # mix_idx = np.where((label>=28) & (label<=75))[0]
        # scores = {'FID': str(fid.compute().item()),
        #           'L1': str(np.mean(l1)),
        #           'L1 std': str(np.std(l1)),
        #           'SSIM': str(np.mean(ssim)),
        #           'SSIM std': str(np.std(ssim)),
                  
        #           't0_filter SSIM': str(np.mean(ssim[t0_filter])),
        #           't0_filter SSIM std': str(np.std(ssim[t0_filter])),
        #           't1_filter SSIM': str(np.mean(ssim[t1_filter])),
        #           't1_filter SSIM std': str(np.std(ssim[t1_filter])),
        #           't2_filter SSIM': str(np.mean(ssim[t2_filter])),
        #           't2_filter SSIM std': str(np.std(ssim[t2_filter])),
                  
        #           'LPIPS': str(np.mean(lpips)),
        #           'LPIPS std': str(np.std(lpips)),
                  
        #           't0_filter LPIPS': str(np.mean(lpips[t0_filter])),
        #           't0_filter LPIPS std': str(np.std(lpips[t0_filter])),
        #           't1_filter LPIPS': str(np.mean(lpips[t1_filter])),
        #           't1_filter LPIPS std': str(np.std(lpips[t1_filter])),
        #           't2_filter LPIPS': str(np.mean(lpips[t2_filter])),
        #           't2_filter LPIPS std': str(np.std(lpips[t2_filter])),
                  
        #           'PSNR': str(np.mean(psnr)),
        #           'PSNR std': str(np.std(psnr)),
        #           'BM ME': str(np.mean(biomass_diff, axis=0)),
        #           'BM ME std': str(np.std(biomass_diff, axis=0)),
        #           'BM MAE': str(np.mean(abs(biomass_diff), axis=0)),
        #           'BM MAE std': str(np.std(abs(biomass_diff), axis=0)),
                  
        #           't0_filter BM ME': str(np.mean(biomass_diff[t0_filter], axis=0)),
        #           't0_filter BM ME std': str(np.std(biomass_diff[t0_filter], axis=0)),
        #           't0_filter BM MAE': str(np.mean(abs(biomass_diff[t0_filter]), axis=0)),
        #           't0_filter BM MAE std': str(np.std(abs(biomass_diff[t0_filter]), axis=0)),
        #           't1_filter BM ME': str(np.mean(biomass_diff[t1_filter], axis=0)),
        #           't1_filter BM ME std': str(np.std(biomass_diff[t1_filter], axis=0)),
        #           't1_filter BM MAE': str(np.mean(abs(biomass_diff[t1_filter]), axis=0)),
        #           't1_filter BM MAE std': str(np.std(abs(biomass_diff[t1_filter]), axis=0)),
        #           't2_filter BM ME': str(np.mean(biomass_diff[t2_filter], axis=0)),
        #           't2_filter BM ME std': str(np.std(biomass_diff[t2_filter], axis=0)),
        #           't2_filter BM MAE': str(np.mean(abs(biomass_diff[t2_filter]), axis=0)),
        #           't2_filter BM MAE std': str(np.std(abs(biomass_diff[t2_filter]), axis=0)),
                  
        #           'Mix BM ME': str(np.mean(biomass_diff[mix_idx], axis=0)),
        #           'Mix BM ME std': str(np.std(biomass_diff[mix_idx], axis=0)),
        #           'Mix BM MAE': str(np.mean(abs(biomass_diff[mix_idx]), axis=0)),
        #           'Mix BM MAE std': str(np.std(abs(biomass_diff[mix_idx]), axis=0)),
        #           }

        # with open(os.path.join(plot_dir,'scores.yaml'), 'w') as file:
        #     yaml.dump(scores, file)   
            
            
        
        #%% calculate metrics
        
        change_if = []
        change_if.append(torch.tensor((1.5,0.5)))
        change_if.append(torch.tensor((1.4,0.6)))
        change_if.append(torch.tensor((1.3,0.7)))
        change_if.append(torch.tensor((1.2,0.8)))
        change_if.append(torch.tensor((1.1,0.9)))
        change_if.append(torch.tensor((1.0,1.0)))
        change_if.append(torch.tensor((0.9,1.1)))
        change_if.append(torch.tensor((0.8,1.2)))
        change_if.append(torch.tensor((0.7,1.3)))
        change_if.append(torch.tensor((0.6,1.4)))
        change_if.append(torch.tensor((0.5,1.5)))
        
        all_cif_mae = []
        all_cif_me =[]
        
        for cif in change_if:

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
            biomass_gen = []
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
                    if_in = batch['biomass_1']
                    cls_in = batch['label']
                    
                    # # img_ref, t_ref, if_ref, cls_ref
                    img_ref = batch['img_2'].to(cfg['device'])
                    t_ref = batch['time_2']
                    if_ref = batch['biomass_2']*cif
                    if_ref[if_ref>1]=1 # set values to zero if 
                    cls_ref = batch['label']
                    
                    # # save times to list
                    t_gen.append(t_ref)
                    t_diff.append(t_ref-t_in)
        
                    # # run model
                    img_gen = model(img_in=img_in,t_in=t_in,t_ref=t_ref,if_in=if_in,if_ref=if_ref,cls_in=cls_in,cls_ref=cls_ref)
                    
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
            
            all_cif_mae.append(np.expand_dims(np.mean(abs(biomass_diff[mix_idx]), axis=0), axis=1))
            all_cif_me.append(np.expand_dims(np.mean(biomass_diff[mix_idx], axis=0), axis=1))
    
            with open(os.path.join(plot_dir,('scores_cif_'+str(round((cif[0]).item(),2))+'_'+str(round((cif[1]).item(),2))+'.yaml')), 'w') as file:
                yaml.dump(scores, file)
            
        #%%
        plot_dir = utils.make_folder(cfg['pred_dir'],(prfx[count]+'cif_bm'))
        
        all_cif_mae = np.concatenate(all_cif_mae, axis=1)
        all_cif_me = np.concatenate(all_cif_me, axis=1)
        
        x = [-50,-40,-30,-20,-10,0,10,20,30,40,50]
        x_tick_labels = ['50:150','60:140','70:130','80:120','90:110','100:100','110:90','120:80','130:70','140:60','150:50']
    
        # # plot
        axis_thickness = 0.6
        fig, axs = plt.subplots(figsize=(cfg['figure_width']*2, cfg['figure_height']), dpi=cfg['plot_dpi'])
        axs.plot(x,all_cif_mae[1,:],c='#95A626',label='$\mathrm{MAE}_{\mathrm{SW}}$')
        axs.plot(x,all_cif_mae[0,:],c='#26A695',label='$\mathrm{MAE}_{\mathrm{FB}}$')

        axs.plot(x,all_cif_me[1,:],linestyle='--',c='#95A626',label='$\mathrm{ME}_{\mathrm{SW}}$ (gen-ref)')
        axs.plot(x,all_cif_me[0,:],linestyle='--',c='#26A695',label='$\mathrm{ME}_{\mathrm{FB}}$ (gen-ref)')

        axs.legend(prop={'size': 7}, loc='lower left', ncol=2)
        axs.set_xticks(x)
        axs.set_xticklabels(x_tick_labels, rotation=90)
        axs.set_xlabel('Biomass-Ratio of SW:FB [\%]')
        axs.set_ylabel('Error [t/ha]')
        axs.set_xlim(-55, 55)
        axs.spines['right'].set_linewidth(axis_thickness)
        axs.spines['left'].set_linewidth(axis_thickness)
        axs.spines['top'].set_linewidth(axis_thickness)
        axs.spines['bottom'].set_linewidth(axis_thickness)
        axs.tick_params(axis='both', which='both', width=axis_thickness)
        axs.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)        
        plt.savefig(os.path.join(plot_dir,'cif_bm.pdf'), dpi=cfg['plot_dpi'], bbox_inches='tight')
        
        
    
        #%% generate random imgs
        # max_plots = 10
        # n_imgs = 8 # per plot, needs to be <= batch_size
        # plot_dir = utils.make_folder(cfg['pred_dir'],(prfx[count]+'gen_rand'))
        # for i_batch, batch in enumerate(dataloader):
        #     if i_batch==max_plots:
        #         break  
            
        #     with torch.no_grad():
        #         # # img_in, t_in, if_in
        #         img_in = batch['img_1']
        #         t_in = batch['time_1']
        #         if_in = batch['biomass_1']
        #         cls_in = batch['label']
                
        #         # # img_ref, t_ref, if_ref
        #         img_ref = batch['img_2']
        #         t_ref = batch['time_2']
        #         if_ref = batch['biomass_2']
        #         cls_ref = batch['label']
                
        #         # # generate z
        #         z = torch.Tensor(np.random.normal(0, 1, (img_in.shape[0],cfg['dim_z'])))

        #         # # run model
        #         # # random generation
        #         img_gen_0 = model(img_in=img_in,t_in=t_in,t_ref=t_ref,if_in=if_in,if_ref=if_ref,cls_in=cls_in,cls_ref=cls_ref,z=z)
                
        #         # # changed cls (input and output)
        #         cls_change = 28 # first mix class
        #         # cls_in_change = (torch.ones(img_in.shape[0])*cls_change).type(torch.int32)
        #         cls_ref_change = (torch.ones(img_in.shape[0])*cls_change).type(torch.int32)
        #         # img_gen_1 = model(img_in=img_in,t_in=t_in,t_ref=t_ref,if_in=if_in,if_ref=if_ref,cls_in=cls_in_change,cls_ref=cls_ref_change,z=z)
                
        #         # # changed cls (only output)
        #         img_gen_2 = model(img_in=img_in,t_in=t_in,t_ref=t_ref,if_in=if_in,if_ref=if_ref,cls_in=cls_in,cls_ref=cls_ref_change,z=z)
                
        #         # # change if to more wheat
        #         if_in_change = if_in+0.2
        #         if_in_change[:,0] = 0
        #         if_ref_change = if_ref+0.2
        #         if_ref_change[:,0] = 0
        #         img_gen_3 = model(img_in=img_in,t_in=t_in,t_ref=t_ref,if_in=if_in_change,if_ref=if_ref_change,cls_in=cls_in,cls_ref=cls_ref,z=z)
                
        #         # # stack images
        #         img_combo = torch.cat((img_in,img_gen_0.to(img_in.device),img_gen_2.to(img_in.device),img_gen_3.to(img_in.device),img_ref),dim=2)
        #         # # build grid
        #         grid = make_grid(cfg['deNorm'](img_combo[:n_imgs,:]))
        #         img = cfg['toPIL'](grid)
                
        #         # # plot
        #         fig, axs = plt.subplots()
        #         axs.imshow(img)
        #         axs.set_title('in-gen-ref ' + str((t_ref-t_in)[:n_imgs].tolist()))
        #         axs.set_axis_off()
        #         plt.savefig(os.path.join(plot_dir,'gen_rand_'+str(i_batch)), dpi=cfg['plot_dpi'], bbox_inches='tight')
        #         plt.close(fig)

            
        
        #%% generate imgs with fixed img_in and z while iteratively increasing t
        # # # # -> check plant consistency over time
        # max_plots=5
        # t_start=21
        # t_end=39
        # t_ref = torch.arange(t_start, t_end)
        # n_imgs = t_ref.shape[0]
        
        # plot_dir = utils.make_folder(cfg['pred_dir'],(prfx[count]+'gen_fixed_in_z'))
        # for i_batch, batch in enumerate(dataloader):
        #     if i_batch==max_plots:
        #         break  
        #     # # generate z
        #     z = torch.Tensor(np.random.normal(0, 1, (1,cfg['dim_z'])))
        #     z = z.repeat(n_imgs,1)
            
        #     with torch.no_grad():
        #         # # img_in pick first image of batch and use it for all others
        #         img_in = batch['img_1'][0, :].unsqueeze(dim=0).repeat(n_imgs,1,1,1)
        #         t_in = batch['time_1'][0].repeat(n_imgs)
                                
        #         img_pred = model(img_in=img_in,t_in=t_in,t_ref=t_ref,z=z)
        #         grid = make_grid(cfg['deNorm'](torch.cat((img_in[0,:].unsqueeze(dim=0).to(img_pred.device),img_pred), dim=0)), nrow=6)
        #         img = cfg['toPIL'](grid)
                
        #         fig, axs = plt.subplots()
        #         axs.imshow(img)
        #         axs.set_axis_off()
        #         plt.savefig(os.path.join(plot_dir,'gen_fixed_in_z_'+str(i_batch)), dpi=cfg['plot_dpi'], bbox_inches='tight')
        #         plt.close(fig)


    #%% eval qual plots dataset
    # plot_dir = utils.make_folder(cfg['pred_dir'],('eval_gen_fixed_in_z'))
    # batch = eval_dataset[0]
    # n_imgs = batch['img'].shape[0]
    # n_runs = 10 # for z

    # for in_idx in range(0,n_imgs):
    #     img_in = batch['img'][in_idx,:].unsqueeze(dim=0).repeat(n_imgs,1,1,1)
    #     t_in = torch.tensor(batch['time'])[in_idx].repeat(n_imgs)
    #     t_ref = torch.tensor(batch['time'])
    #     img_ref = batch['img'].detach().clone()
    #     img_pred = torch.empty((n_runs,n_imgs,3,cfg['img_size'],cfg['img_size']))

    #     for j in range(0,n_runs):
    #         # # generate z
    #         z = torch.Tensor(np.random.normal(0, 1, (1,cfg['dim_z'])))
    #         z = z.repeat(n_imgs,1)
            
    #         with torch.no_grad():
    #             img_pred[j,:] = model(img_in=img_in,t_in=t_in,t_ref=t_ref,z=z)
        
    #     # # use img of first run [0] for visualisation and make them to grid
    #     grid = make_grid(cfg['deNorm'](img_pred[0,:]), nrow=n_imgs)
    #     img_pred_grid = cfg['toPIL'](grid)
        
    #     # # compute std over all runs and then mean over all channels and make them to grid
    #     img_pred_std = torch.mean(torch.std(img_pred, axis=0), axis=1)
    #     grid = make_grid(torch.unsqueeze(img_pred_std,dim=1), nrow=n_imgs)
    #     img_std_grid = cfg['toPIL'](grid[0,:]) # grid always creates a 3-channel map, but all 3 channels are the same, as input is BW-img -> just use first channel grid[0,:] -> so that we can apply cmap in the end

    #     # # metrics (before manipulating refs with colored boxes :-D)
    #     l1 = torch.mean(loss_l1(img_ref, img_pred[0,:]), [1,2,3], True).cpu()
    #     ssim = loss_ssim(img_ref, img_pred[0,:]).cpu()
    #     lpips = []
    #     for k in range(img_ref.shape[0]):
    #         lpips.append(loss_lpips(img_ref[k,:].unsqueeze(dim=0), img_pred[0,k,:].unsqueeze(dim=0)).cpu().detach().item())
        
        
    #     # # print blue box around in_img of img_ref and make them to grid
    #     img_ref[in_idx,2,0:10,:]=1
    #     img_ref[in_idx,2,246:256,:]=1
    #     img_ref[in_idx,2,:,0:10]=1
    #     img_ref[in_idx,2,:,246:256]=1
    #     img_ref[in_idx,0:2,0:10,:]=0
    #     img_ref[in_idx,0:2,246:256,:]=0
    #     img_ref[in_idx,0:2,:,0:10]=0
    #     img_ref[in_idx,0:2,:,246:256]=0
    #     grid = make_grid(cfg['deNorm'](img_ref), nrow=n_imgs)
    #     img_ref_grid = cfg['toPIL'](grid)
        
        
    #     # # plot everythin in one figure
    #     fig, axs = plt.subplots(6,1)
        
    #     axs[0].imshow(img_ref_grid)
    #     # axs[0].set_axis_off()
    #     axs[0].set_yticklabels([])
    #     axs[0].set_xticklabels([])
    #     axs[0].set_ylabel("ref")
        
    #     axs[1].imshow(img_pred_grid)
    #     # axs[1].set_axis_off()
    #     axs[1].set_yticklabels([])
    #     axs[1].set_xticklabels([])
    #     axs[1].set_ylabel("gen")
        
    #     axs[2].imshow(img_std_grid,cmap=plt.cm.Blues)#,vmin=0,vmax=10)
    #     # axs[2].set_axis_off()
    #     axs[2].set_yticklabels([])
    #     axs[2].set_xticklabels([])
    #     axs[2].set_ylabel("std")
        
    #     axs[3].plot(t_ref,l1.squeeze())
    #     axs[3].set_ylim(0, 0.2)
    #     axs[3].set_xlim(min(t_ref), max(t_ref))
    #     axs[3].set_xticklabels([])
    #     axs[3].set_ylabel("L1")
    #     axs[3].grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
        
    #     axs[4].plot(t_ref,lpips)
    #     axs[4].set_ylim(0, 0.5)
    #     axs[4].set_xlim(min(t_ref), max(t_ref))
    #     axs[4].set_xticklabels([])
    #     axs[4].set_ylabel("LPIPS")
    #     axs[4].grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
        
    #     axs[5].plot(t_ref,ssim.squeeze())
    #     axs[5].set_ylim(0.5, 1)
    #     axs[5].set_xlim(min(t_ref), max(t_ref))
    #     axs[5].set_ylabel("SSIM") # MS-SSIM is too long
    #     axs[5].set_xlabel("days after sowing [DAS]") # days after seeding/sowing
    #     axs[5].grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

    #     # plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.5)
    #     # plt.subplots_adjust(wspace=-0.5, hspace=-0.5)
    #     plt.savefig(os.path.join(plot_dir,'gen_fixed_in_z_'+str(t_ref[in_idx].item())), dpi=cfg['plot_dpi']+200, bbox_inches='tight')
    #     plt.close(fig)
        