"""
===============================================================================
Train CGAN with img, t, and if as conditions
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
import wandb
from utils import utils
from pytorch_lightning.callbacks import Callback

from configs.config_train_img_t_if import cfg

from datasets.mc_datamodule import MixCropDataModule, MixCrop2ImagesDataModule, MixCrop2Images2DatesDataModule
from models.wgangp_img_t_if_plm import WGANGPModel_img_t_if


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


if __name__ == '__main__':
        
    
    #%% write cfg.yaml to exp_dir
    with io.open(os.path.join(cfg['exp_dir'], 'cfg_main.yaml'), 'w', encoding='utf8') as outfile:
        yaml.dump(cfg, outfile, default_flow_style=False, allow_unicode=True)
    
    
    #%% dataModule
    dataModule = MixCrop2ImagesDataModule(cfg['img_dir'], cfg['info_tab_path'], cfg['wheat_target_path'], cfg['bean_target_path'], cfg['mix_target_path'], cfg['wheat_treatment_path'], cfg['bean_treatment_path'], cfg['mix_treatment_path'], cfg['data_name'], cfg['data_time'], cfg['batch_size'], cfg['n_workers'], cfg['transform_train'], cfg['transform_test'], target_type=cfg['target_type'], target_transform=cfg['target_transform'], in_memory=cfg['in_memory'], val_test_shuffle=cfg['val_test_shuffle'])

    # setup dataModule
    dataModule.prepare_data()
    dataModule.setup()
    
    # show dim and len of different data subsets
    print('---Some Training Stats---')
    print('Input dims:', dataModule.data_dims)
    print('#Traindata:', len(dataModule.train_dataloader().dataset))
    print('#Valdata:', len(dataModule.val_dataloader().dataset))
    print('#Testdata:', len(dataModule.test_dataloader().dataset))    
    
    # write dataModule params
    with open(os.path.join(cfg['exp_dir'], 'hparams_data.yml'), 'w') as outfile:
        yaml.dump(dataModule.params, outfile, default_flow_style=False, allow_unicode=True)
    
    
    #%% visualize training sample
    # show x sample from train set (it is always the first image of the batch)
    max_plots = 5
    train_dataloader = dataModule.train_dataloader()
    for i_batch, batch in enumerate(train_dataloader):
        if i_batch==max_plots:
            break
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(cfg['toPIL'](cfg['deNorm'](np.squeeze(batch['img_1'][0,:,:,:]))))
        axs[0].set_title(str(batch['time_1'][0].numpy().item())+' '+str(np.round(batch['biomass_1'][0][0].numpy().item()))+' '+str(np.round(batch['biomass_1'][0][1].numpy().item())))
        axs[1].imshow(cfg['toPIL'](cfg['deNorm'](np.squeeze(batch['img_2'][0,:,:,:]))))
        axs[1].set_title(str(batch['time_2'][0].numpy().item())+' '+str(np.round(batch['biomass_2'][0][0].numpy().item()))+' '+str(np.round(batch['biomass_2'][0][1].numpy().item())))
        #plt.close(fig)
    
    
    #%% build a model        
    if cfg['use_model'] == 'gan':
        model = GANModel_img_t_if(dataModule.data_dims,cfg['g_e_net'],cfg['g_e_net_pretrained'],cfg['g_d_net'],cfg['g_d_net_pretrained'],cfg['dim_z'],cfg['dim_w'],cfg['dim_img'],cfg['dim_t'],cfg['dim_if'],cfg['z_fusion_type'],cfg['t_fusion_type'],cfg['data_time'],cfg['d_net'],cfg['d_net_pretrained'],cfg['d_transforms'],cfg['p_d_transforms'],cfg['lr'],cfg['losses_w'],cfg['final_actvn'])
    elif cfg['use_model'] == 'wgangp':
        model = WGANGPModel_img_t_if(dataModule.data_dims,cfg['g_e_net'],cfg['g_e_net_pretrained'],cfg['g_d_net'],cfg['g_d_net_pretrained'],cfg['dim_z'],cfg['dim_w'],cfg['dim_img'],cfg['dim_t'],cfg['dim_if'],cfg['z_fusion_type'],cfg['t_fusion_type'],cfg['data_time'],cfg['d_net'],cfg['d_net_pretrained'],cfg['d_transforms'],cfg['p_d_transforms'],cfg['lr'],cfg['losses_w'],cfg['final_actvn'])
    else:
        print('ERROR: FALSE MODEL SPECIFIED!')
    print(model.hparams)
    
    
    #%% Log gradients and some generated images
    # # Gradients
    cfg['logger'].watch(model,log_graph='False')
    
    # # generated images
    class LogPredictionsCallback(Callback):
        
        def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
            """Called when the validation batch ends."""
     
            # # `outputs` come from `LightningModule.validation_step`
            # # which corresponds to our model predictions in this case
                     
            # # Let's log 4 sample image predictions from first batch
            if batch_idx == 0 and pl_module.current_epoch % 10 == 0:
                grid = torchvision.utils.make_grid(cfg['deNorm'](outputs[0][:4,:]))
                # # Option 1: log images with `WandbLogger.log_image`
                cfg['logger'].log_image(key='sample_images', images=[grid], caption=['in-gen-ref ' + str(outputs[1][:4].tolist())])
                
    log_predictions_callback = LogPredictionsCallback()
    cfg['callbacks'] += [log_predictions_callback]
    
    
    #%% training
    # # Build a trainer from train parameters, callbacks, and logger
    trainer = pl.Trainer(
        max_epochs=cfg['max_epochs'], 
        gpus=cfg['gpus'],
        callbacks=cfg['callbacks'],
        logger=[cfg['logger']],
        precision=cfg['precision'],
        fast_dev_run=cfg['fast_dev_run'], 
        limit_train_batches=cfg['limit_train_batches'],
        limit_val_batches=cfg['limit_val_batches'],
        limit_test_batches=cfg['limit_test_batches'],
    )
    
    # # train
    start_time = time.time()
    trainer.fit(model, dataModule,ckpt_path=cfg['ckpt_path_resume'])
    print('Training finished. Elapsed Time:', str(round((time.time()-start_time)/60,2)), 'min')
    wandb.finish()
    
    
    #%% test  
    if cfg['run_test']:
        trainer.test(verbose=False)
    
    
    #%% plotting
    if not cfg['run_plots']:
        sys.exit()
        
        
    #%% load model from best checkpoint if available otherwise last checkpoint is loaded automatically
    # # or uncomment last_model_path manually
    ckpt_path = trainer.checkpoint_callback.best_model_path
    # ckpt_path = trainer.checkpoint_callback.last_model_path
    print('ckpt_path: ', ckpt_path)
    
    if cfg['use_model'] == 'gan':
        model = GANModel_img_t_if.load_from_checkpoint(ckpt_path)
    elif cfg['use_model'] == 'wgangp':
        model = WGANGPModel_img_t_if.load_from_checkpoint(ckpt_path)
        
    # # set to eval mode
    model.eval()    
    # # sent model to device
    model.to(cfg['device'])