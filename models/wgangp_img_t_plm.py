"""
WGAN-GP with img and t
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import timm
import random
# import copy

from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from models.noise_mapping import MappingNetwork
from models.encoder_networks import ResNet18EncoderCbn, ResNet18EncoderCbnNoPool
from models.decoder_networks import ResNet18DecoderCbnNoPool
from models.lightweight_gan_generator import LWGenerator, LWGeneratorCBN
from models.discriminator_networks import Layer3Discriminator_img_t, Layer5Discriminator_img_t

from models.embedding import TimeEmbedding


class WGANGPModel_img_t(pl.LightningModule):
    def __init__(self, data_shape, g_e_net, g_e_net_pretrained, g_d_net, g_d_net_pretrained, dim_z, dim_w, dim_img, dim_t, z_fusion_type, t_fusion_type, data_time, d_net, d_net_pretrained, d_transforms, p_d_transforms, lr, losses_w, final_actvn, b1=0.0, b2=0.9, n_critic=5, lambda_gp=10):
        
        '''
        Parameters
        ----------
        data_shape : size object of torch module.
            dimension: [C,W,H].
        g_e_net : str
            Name of geneartor encoder network
        g_e_net_pretrained : bool
            Use pretrained geneartor encoder network?        
        g_d_net : str
            Name of geneartor decoder network
        g_d_net_pretrained : bool
            Use pretrained geneartor decoder network?
        dim_z : int
            dimension stochasticity
        dim_w : int
            dimension of mapped stochasticity (or None if no mapping)
        dim_img : int
            channel dim of img embedding
        dim_t : int
            dimension of pos_enc of t
        z_fusion_type : str
            fusion type of z resp. w and img embedding
        t_fusion_type : str
            fusion type of t and img emebdding
        data_time : dict
            time dict of data, needed for z fusion type wadd
        d_net : str
            Name of discriminator network
        d_net_pretrained : bool
            Use pretrained discriminator?
        d_transforms : list
            list of augmentations applied only to the discriminator
        p_d_transforms : int
            probability for each d_transforms to be applied
        lr : float
            learning rate.
        losses_w : dict
            contains weights of losses
        final_actvn : str
            indicates the final image activation

        Returns
        -------
        None.

        '''
        super().__init__()
        self.save_hyperparameters()
        self.data_shape = data_shape
        self.c = data_shape[0]
        self.w = data_shape[1]
        self.h = data_shape[2]
        self.g_e_net = g_e_net
        self.g_e_net_pretrained = g_e_net_pretrained
        self.g_d_net = g_d_net
        self.g_d_net_pretrained = g_d_net_pretrained
        self.dim_z = dim_z
        self.dim_w = dim_w
        self.dim_img = dim_img
        self.dim_t = dim_t
        self.z_fusion_type = z_fusion_type
        self.t_fusion_type = t_fusion_type
        self.data_time = data_time
        self.d_net = d_net
        self.d_net_pretrained = d_net_pretrained
        self.d_transforms = d_transforms
        self.p_d_transforms = p_d_transforms
        self.lr = lr
        self.losses_w = losses_w
        self.final_actvn = final_actvn
        # WGAN gradient penalty Parameter for optimization
        self.b1 = b1
        self.b2 = b2
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp
        
        # # Generator Encoder
        if g_e_net=='res18':
            self.g_e = timm.create_model('resnet18', pretrained=g_e_net_pretrained)
            self.g_e = torch.nn.Sequential(*list(self.g_e.children())[:-1])
        elif g_e_net=='res50':
            self.g_e = timm.create_model('resnet50', pretrained=g_e_net_pretrained)
            self.g_e = torch.nn.Sequential(*list(self.g_e.children())[:-1])
        elif g_e_net=='res18_CBN':
            self.g_e = ResNet18EncoderCbn(self.c, self.dim_t)
        elif g_e_net=='res18_CBN_noPool':
            self.g_e = ResNet18EncoderCbnNoPool(self.c, self.dim_t)
        else:
            print('Wrong g_e_net')
        print('Generator encoder: Total params:', sum(p.numel() for p in self.g_e.parameters()))
        
        # # Noise mapping
        if self.dim_w:
            self.noise_mapping = MappingNetwork(self.dim_z, self.dim_z+int((self.dim_w-self.dim_z)/2), self.dim_w, 3)
            self.use_noise_mapping = True
        else:
            self.noise_mapping = lambda x: x
            self.dim_w = self.dim_z
            self.use_noise_mapping = False
        
        # # Generator Decoder
        if g_d_net=='lightweight':
            if self.z_fusion_type == 'cat':
                self.g_d = LWGenerator(self.w, latent_dim=self.dim_img+self.dim_w+self.dim_t*2)
            else: # None, 'add', 'wadd'
                self.g_d = LWGenerator(self.w, latent_dim=self.dim_img+self.dim_t*2)
        elif g_d_net=='lightweight_CBN':
            if self.z_fusion_type == 'cat':
                self.g_d = LWGeneratorCBN(self.w, self.dim_t, latent_dim=self.dim_img+self.dim_w)
            else: # None, 'add', 'wadd'
                self.g_d = LWGeneratorCBN(self.w, self.dim_t, latent_dim=self.dim_img)
        elif g_d_net=='res18_CBN_noPool':
            if self.z_fusion_type == 'cat':
                self.g_d = ResNet18DecoderCbnNoPool(self.dim_t, cat_channels=self.dim_w)
            else: # None, 'add', 'wadd'
                self.g_d = ResNet18DecoderCbnNoPool(self.dim_t)
        else:
            print('Wrong g_d_net')
        print('Generator decoder: Total params:', sum(p.numel() for p in self.g_d.parameters()))
        
        # # Discriminator
        if d_net == 'layer3_img_t':
            self.d = Layer3Discriminator_img_t(self.c*2, self.dim_t, norm_layer=nn.InstanceNorm2d)
        elif d_net == 'layer5_img_t':
            self.d = Layer5Discriminator_img_t(self.c*2, self.dim_t, norm_layer=nn.InstanceNorm2d)
        else:
            print('Wrong d_net')
        print('Discriminator: Total params:', sum(p.numel() for p in self.d.parameters()))
        
        # # positional encoding
        self.t_emb_g = TimeEmbedding(self.dim_t, self.dim_t)
        
        # # activations
        if self.final_actvn == 'relu':
            self.actvn = torch.nn.ReLU()
        elif self.final_actvn == 'sigmoid':
            self.actvn = torch.nn.Sigmoid()
        elif self.final_actvn == 'tanh':
            self.actvn=torch.nn.Tanh()
        else:
            print('Wrong final_actvn.')
            
        # # Losses
        self.loss_l1 = torch.nn.L1Loss()
        self.loss_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
        self.loss_lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True)
    
        # # example input: this allows the trainer to show input and output sizes in the report (12 is just a sample batch size)
        self.example_input_array = {}
    
    
    def compute_gradient_penalty(self, real, fake, t_1, t_2):
        """Calculates the gradient penalty loss for WGAN GP"""
        # # Random weight term for interpolation between real and fake samples
        alpha = torch.Tensor(np.random.random((real.size(0), 1, 1, 1))).to(self.device)
        
        # # Get random interpolation between real and fake samples
        interpolates = (alpha * real + ((1 - alpha) * fake)).requires_grad_(True).to(self.device)
        
        # # Calculate critic scores of interpolated samples
        d_interpolates = self.d(interpolates,t_1,t_2)
            
        # # Calculate gradients of critic scores with respect to interpolated samples
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        # # Calculate gradient penalty
        gradients = gradients.view(gradients.size(0), -1).to(self.device)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    
    def configure_optimizers(self):
        g_parameters = []
        g_parameters += list(self.g_e.parameters())
        g_parameters += list(self.g_d.parameters())
        g_parameters += list(self.t_emb_g.parameters())
        if self.use_noise_mapping:
            g_parameters += list(self.noise_mapping.parameters())
        optimizer_g = torch.optim.Adam(g_parameters, lr=self.hparams.lr)
        optimizer_d = torch.optim.Adam(self.d.parameters(), lr=self.hparams.lr)

        return [optimizer_g, optimizer_d]

    
    def _forward(self, img_in, t_in, t_ref, z=None):
        if self.g_e_net=='res18_CBN' or self.g_e_net=='res18_CBN_noPool':
            x = self.g_e(img_in, self.t_emb_g(t_in))
        else:
            x = self.g_e(img_in)
        
        # # create z and map to w
        if z == None:
            z = torch.Tensor(np.random.normal(0, 1, (img_in.shape[0],self.dim_z))).to(self.device)
        else:
            z = z.to(self.device)
        w = self.noise_mapping(z)
        w = w.view(w.shape[0], w.shape[1], 1, 1)
        w = w.repeat(1,1,x.shape[2], x.shape[3])
        
        # # fuse x and w
        if self.z_fusion_type == 'add':
            x = x+w
        elif self.z_fusion_type == 'wadd':
            abs_period = (self.data_time['time_end']-self.data_time['time_start']).days
            rel_diff = (abs(t_ref-t_in)/abs_period).unsqueeze(dim=1).view(-1,1,1,1).repeat(1,x.shape[1],1,1)
            x = (1-rel_diff)*x + rel_diff*w
        elif self.z_fusion_type == 'cat':
            x = torch.cat((x,w), dim=1)
        
        # # Decoding
        if self.g_d_net=='lightweight':
            t_emb = torch.cat((self.t_emb_g(t_in),self.t_emb_g(t_ref)), dim=1)
            t_emb = t_emb.view(t_emb.shape[0], t_emb.shape[1], 1, 1)
            x = self.g_d(torch.cat((x,t_emb), dim=1))
        elif self.g_d_net=='lightweight_CBN' or self.g_d_net=='res18_CBN_noPool':
            x = self.g_d(x,self.t_emb_g(t_ref))
        return x

    
    def training_step(self, batch, batch_idx, optimizer_idx):
        # # img_in, t_in
        img_in = batch['img_1']
        t_in = batch['time_1']
        
        # # img_ref, t_ref
        img_ref = batch['img_2']
        t_ref = batch['time_2']
        
        # # run forward pass
        logits_gen = self._forward(img_in,t_in,t_ref)
        # # activate output 
        img_gen = self.actvn(logits_gen)
        
        # # discriminator transform 
        prob = random.uniform(0,1)
        seed = np.random.randint(245985462)
        if prob < self.p_d_transforms:
            # # Apply transformations separately for each sample in the stack
            torch.manual_seed(seed)
            random.seed(seed)
            img_gen = torchvision.transforms.Lambda(lambda x: torch.stack([self.d_transforms(x_) for x_ in img_gen]))(img_gen)
            torch.manual_seed(seed)
            random.seed(seed)
            img_ref = torchvision.transforms.Lambda(lambda x: torch.stack([self.d_transforms(x_) for x_ in img_ref]))(img_ref)
        
        # # train generator
        if optimizer_idx == 0:
            # # run discriminator
            fake_validity = self.d(torch.cat((img_in,img_gen),dim=1),t_in,t_ref)
                
            # # compute gan loss
            g_loss = -torch.mean(fake_validity)
            self.log('loss_g', g_loss)
            # # compute l1_loss
            loss_l1 = self.loss_l1(img_gen, img_ref)
            self.log('loss_l1', loss_l1)
            # # # compute ssim loss
            # loss_ssim = 1-self.loss_ssim(img_gen, img_ref)
            # self.log("loss_ssim", loss_ssim)
            # # # compute LPIPS
            # loss_lpips = self.loss_lpips(img_gen, img_ref)
            # self.log("loss_lpips", loss_lpips)
            
            # # combine losses
            # loss = self.losses_w['weight_adv'] * g_loss + self.losses_w['weight_l1'] * loss_l1 + self.losses_w['weight_ssim'] * loss_ssim + self.losses_w['weight_lpips'] * loss_lpips
            loss = self.losses_w['weight_adv'] * g_loss + self.losses_w['weight_l1'] * loss_l1
        
            return loss
        
        # # train discriminator    
        elif optimizer_idx == 1:
            # # run discriminator and gradient penalty
            fake_validity = self.d(torch.cat((img_in,img_gen),dim=1),t_in,t_ref)
            real_validity = self.d(torch.cat((img_in,img_ref),dim=1),t_in,t_ref)
            gradient_penalty = self.compute_gradient_penalty(torch.cat((img_in,img_ref),dim=1),torch.cat((img_in,img_gen),dim=1),t_in,t_ref)
            self.log('gradient_penalty',gradient_penalty)
            
            # # Wasserstein distance (wd)
            wd = -torch.mean(real_validity) + torch.mean(fake_validity)
            self.log('wd', wd)
            
            # # Discriminator loss
            d_loss = wd + self.lambda_gp * gradient_penalty
            self.log('loss_d', d_loss)
            
            return d_loss
    
    
    def validation_step(self, batch, batch_idx):   
        # # img_in, t_in
        img_in = batch['img_1']
        t_in = batch['time_1']
        
        # # img_ref, t_ref
        img_ref = batch['img_2']
        t_ref = batch['time_2']
        
        # # run forward pass
        logits_gen = self._forward(img_in,t_in,t_ref)
        # # activate output 
        img_gen = self.actvn(logits_gen)
        
        # # run discriminator
        fake_validity = self.d(torch.cat((img_in,img_gen),dim=1),t_in,t_ref)
        real_validity = self.d(torch.cat((img_in,img_ref),dim=1),t_in,t_ref)

        # # Wasserstein distance (wd)
        wd = -torch.mean(real_validity) + torch.mean(fake_validity)
        self.log('val_wd', wd)
        
        # # compute l1_loss
        loss_l1 = self.loss_l1(img_gen, img_ref)
        self.log('val_loss_l1', loss_l1)
        # # compute ssim loss
        loss_ssim = 1-self.loss_ssim(img_gen, img_ref)
        self.log("val_loss_ssim", loss_ssim)
        # # compute LPIPS
        loss_lpips = self.loss_lpips(img_gen, img_ref)
        self.log("val_loss_lpips", loss_lpips)
        
        # # save here the metric that should define the best model. Important for ModelChecker: log name: 'val_loss'
        self.log('val_loss', loss_lpips, on_step=False, on_epoch=True)
        
        return torch.cat((img_in,img_gen,img_ref),dim=2), t_ref-t_in

    
    def forward(self, img_in=None, t_in=None, t_ref=None, z=None):
        # # generate one dummy img_in if its None   
        if img_in == None:
            img_in = torch.zeros((1, self.c, self.w, self.h)).to(self.device)
        else:
            img_in = img_in.to(self.device)
            
        n_imgs = img_in.shape[0]   
        
        # # generate t_in     
        if t_in == None:
            t_in = torch.randint(22, 38, (n_imgs,)).to(self.device)
        else:
            t_in = t_in.to(self.device)
            
        # # generate t_ref     
        if t_ref == None:
            t_ref = torch.randint(22, 38, (n_imgs,)).to(self.device)
        else:
            t_ref = t_ref.to(self.device)
            
        # run forward pass
        logits_gen = self._forward(img_in, t_in, t_ref, z=z)
        img_gen = self.actvn(logits_gen)
        
        return img_gen 
