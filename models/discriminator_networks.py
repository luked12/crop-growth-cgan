import torch
import torch.nn as nn
import functools

from models.embedding import Embedding, TimeEmbedding

'''
code reference: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
-> manually adaption (remove use_bias, add affine=True)
'''
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer (nn.BatchNorm2d or nn.InstanceNorm2d)
        """
        super(NLayerDiscriminator, self).__init__()

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                norm_layer(ndf * nf_mult, affine=True),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            norm_layer(ndf * nf_mult, affine=True),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


'''
Reformulated PatchGAN (NLayerDiscriminator)
Layer N ... N refers to the number of downscaling convs (stride=2)
'''
class Layer3Discriminator(nn.Module):
    def __init__(self, channels, hidden_dim=64, norm_layer=nn.BatchNorm2d):
        super(Layer3Discriminator, self).__init__()

        self.channels = channels
        self.hidden_dim = hidden_dim
        self.norm_layer = norm_layer
        
        self.init_block = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            )
        
        self.main_block = nn.Sequential(
            self._block(hidden_dim,   hidden_dim*2, 4, 2, 1, bias=False),
            self._block(hidden_dim*2, hidden_dim*4, 4, 2, 1, bias=False),
            self._block(hidden_dim*4, hidden_dim*8, 4, 2, 1, bias=False),
            )
        
        self.final_block = nn.Sequential(
            self._block(hidden_dim*8, hidden_dim*8, 4, 1, 1, bias=False),
            nn.Conv2d(hidden_dim*8, 1, kernel_size=4, stride=1, padding=0),
            )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            self.norm_layer(out_channels, affine=True),
            nn.LeakyReLU(0.2),
            )
        return block
    
    def forward(self, x):
        x = self.init_block(x)
        x = self.main_block(x)
        x = self.final_block(x)
        return x.view(-1, 1).squeeze(dim=1)
    

'''
Reformulated PatchGAN (NLayerDiscriminator) with included hidden concatenation of t embeddings
Layer N ... N refers to the number of downscaling convs (stride=2)
'''
class Layer3Discriminator_t(nn.Module):
    def __init__(self, channels, dim_t, hidden_dim=64, norm_layer=nn.BatchNorm2d):
        super(Layer3Discriminator_t, self).__init__()

        self.channels = channels
        self.dim_t = dim_t
        self.hidden_dim = hidden_dim
        self.norm_layer = norm_layer
        
        self.init_block = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            )
        
        self.main_block = nn.Sequential(
            self._block(hidden_dim,   hidden_dim*2, 4, 2, 1, bias=False),
            self._block(hidden_dim*2, hidden_dim*4, 4, 2, 1, bias=False),
            self._block(hidden_dim*4, hidden_dim*8, 4, 2, 1, bias=False),
            )
        
        self.time_emb = TimeEmbedding(16*16, self.dim_t)
        
        self.final_block = nn.Sequential(
            self._block(hidden_dim*8+1, hidden_dim*8, 4, 1, 1, bias=False),
            nn.Conv2d(hidden_dim*8, 1, kernel_size=4, stride=1, padding=0),
            )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            self.norm_layer(out_channels, affine=True),
            nn.LeakyReLU(0.2),
            )
        return block
    
    def forward(self, x, t):
        x = self.init_block(x)
        x = self.main_block(x)
        
        # MLP(EMBED(T)) and concat
        t = self.time_emb(t).view(-1, 1, 16, 16)
        x = torch.cat([x, t], dim=1)
        
        x = self.final_block(x)
        return x.view(-1, 1).squeeze(dim=1)
    
    
'''
Reformulated PatchGAN (NLayerDiscriminator) with included hidden concatenation of t embeddings
Layer N ... N refers to the number of downscaling convs (stride=2)
'''
class Layer3Discriminator_img_t(nn.Module):
    def __init__(self, channels, dim_t, hidden_dim=64, norm_layer=nn.BatchNorm2d):
        super(Layer3Discriminator_img_t, self).__init__()

        self.channels = channels
        self.dim_t = dim_t
        self.hidden_dim = hidden_dim
        self.norm_layer = norm_layer
        
        self.init_block = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            )
        
        self.main_block = nn.Sequential(
            self._block(hidden_dim,   hidden_dim*2, 4, 2, 1, bias=False),
            self._block(hidden_dim*2, hidden_dim*4, 4, 2, 1, bias=False),
            self._block(hidden_dim*4, hidden_dim*8, 4, 2, 1, bias=False),
            )
        
        self.time_emb = TimeEmbedding(16*16, self.dim_t)
        
        self.final_block = nn.Sequential(
            self._block(hidden_dim*8+2, hidden_dim*8, 4, 1, 1, bias=False),
            nn.Conv2d(hidden_dim*8, 1, kernel_size=4, stride=1, padding=0),
            )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            self.norm_layer(out_channels, affine=True),
            nn.LeakyReLU(0.2),
            )
        return block
    
    def forward(self, x, t_1, t_2):
        x = self.init_block(x)
        x = self.main_block(x)
        
        # MLP(EMBED(T)) and concat
        t_1 = self.time_emb(t_1).view(-1,1,16,16)
        t_2 = self.time_emb(t_2).view(-1,1,16,16)
        x = torch.cat([x, t_1, t_2], dim=1)
        
        x = self.final_block(x)
        return x.view(-1, 1).squeeze(dim=1)


'''
Reformulated PatchGAN (NLayerDiscriminator) with included hidden concatenation of t embeddings
Layer N ... N refers to the number of downscaling convs (stride=2)
'''
class Layer3Discriminator_img_t_if(nn.Module):
    def __init__(self, channels, dim_t, dim_if, hidden_dim=64, norm_layer=nn.BatchNorm2d):
        super(Layer3Discriminator_img_t_if, self).__init__()

        self.channels = channels
        self.dim_t = dim_t
        self.dim_if = dim_if
        self.hidden_dim = hidden_dim
        self.norm_layer = norm_layer
        
        self.init_block = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            )
        
        self.main_block = nn.Sequential(
            self._block(hidden_dim,   hidden_dim*2, 4, 2, 1, bias=False),
            self._block(hidden_dim*2, hidden_dim*4, 4, 2, 1, bias=False),
            self._block(hidden_dim*4, hidden_dim*8, 4, 2, 1, bias=False),
            )
        
        self.time_emb = TimeEmbedding(16*16, self.dim_t)
        self.if_emb = Embedding(self.dim_if, 16*16)
        
        self.final_block = nn.Sequential(
            self._block(hidden_dim*8+4, hidden_dim*8, 4, 1, 1, bias=False),
            nn.Conv2d(hidden_dim*8, 1, kernel_size=4, stride=1, padding=0),
            )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            self.norm_layer(out_channels, affine=True),
            nn.LeakyReLU(0.2),
            )
        return block
    
    def forward(self, x, t_1, t_2, if_1, if_2):
        x = self.init_block(x)
        x = self.main_block(x)
        
        # MLP(EMBED(T)) and concat
        t_1 = self.time_emb(t_1).view(-1,1,16,16)
        t_2 = self.time_emb(t_2).view(-1,1,16,16)
        if_1 = self.if_emb(if_1).view(-1,1,16,16)
        if_2 = self.if_emb(if_2).view(-1,1,16,16)
        x = torch.cat([x, t_1, t_2, if_1, if_2], dim=1)
        
        x = self.final_block(x)
        return x.view(-1, 1).squeeze(dim=1)



'''
Reformulated PatchGAN (NLayerDiscriminator) with included hidden concatenation of t embeddings
Layer N ... N refers to the number of downscaling convs (stride=2)
'''
class Layer3Discriminator_img_t_cls(nn.Module):
    def __init__(self, channels, dim_t, dim_cls, hidden_dim=64, norm_layer=nn.BatchNorm2d):
        super(Layer3Discriminator_img_t_cls, self).__init__()

        self.channels = channels
        self.dim_t = dim_t
        self.dim_cls = dim_cls
        self.hidden_dim = hidden_dim
        self.norm_layer = norm_layer
        
        self.init_block = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            )
        
        self.main_block = nn.Sequential(
            self._block(hidden_dim,   hidden_dim*2, 4, 2, 1, bias=False),
            self._block(hidden_dim*2, hidden_dim*4, 4, 2, 1, bias=False),
            self._block(hidden_dim*4, hidden_dim*8, 4, 2, 1, bias=False),
            )
        
        self.time_emb = TimeEmbedding(16*16, self.dim_t)
        self.cls_emb = nn.Embedding(self.dim_cls, 16*16)
        
        self.final_block = nn.Sequential(
            self._block(hidden_dim*8+4, hidden_dim*8, 4, 1, 1, bias=False),
            nn.Conv2d(hidden_dim*8, 1, kernel_size=4, stride=1, padding=0),
            )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            self.norm_layer(out_channels, affine=True),
            nn.LeakyReLU(0.2),
            )
        return block
    
    def forward(self, x, t_1, t_2, cls_1, cls_2):
        x = self.init_block(x)
        x = self.main_block(x)
        
        # MLP(EMBED(T)) and concat
        t_1 = self.time_emb(t_1).view(-1,1,16,16)
        t_2 = self.time_emb(t_2).view(-1,1,16,16)
        cls_1 = self.cls_emb(cls_1).view(-1,1,16,16)
        cls_2 = self.cls_emb(cls_2).view(-1,1,16,16)
        x = torch.cat([x, t_1, t_2, cls_1, cls_2], dim=1)
        
        x = self.final_block(x)
        return x.view(-1, 1).squeeze(dim=1)
    

'''
Reformulated PatchGAN (NLayerDiscriminator) with included hidden concatenation of t embeddings
Layer N ... N refers to the number of downscaling convs (stride=2)
'''
class Layer3Discriminator_img_t_if_cls(nn.Module):
    def __init__(self, channels, dim_t, dim_if, dim_cls, hidden_dim=64, norm_layer=nn.BatchNorm2d):
        super(Layer3Discriminator_img_t_if_cls, self).__init__()

        self.channels = channels
        self.dim_t = dim_t
        self.dim_if = dim_if
        self.dim_cls = dim_cls
        self.hidden_dim = hidden_dim
        self.norm_layer = norm_layer
        
        self.init_block = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            )
        
        self.main_block = nn.Sequential(
            self._block(hidden_dim,   hidden_dim*2, 4, 2, 1, bias=False),
            self._block(hidden_dim*2, hidden_dim*4, 4, 2, 1, bias=False),
            self._block(hidden_dim*4, hidden_dim*8, 4, 2, 1, bias=False),
            )
        
        self.time_emb = TimeEmbedding(16*16, self.dim_t)
        self.if_emb = Embedding(self.dim_if, 16*16)
        self.cls_emb = nn.Embedding(self.dim_cls, 16*16)
        
        self.final_block = nn.Sequential(
            self._block(hidden_dim*8+6, hidden_dim*8, 4, 1, 1, bias=False),
            nn.Conv2d(hidden_dim*8, 1, kernel_size=4, stride=1, padding=0),
            )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            self.norm_layer(out_channels, affine=True),
            nn.LeakyReLU(0.2),
            )
        return block
    
    def forward(self, x, t_1, t_2, if_1, if_2, cls_1, cls_2):
        x = self.init_block(x)
        x = self.main_block(x)
        
        # MLP(EMBED(T)) and concat
        t_1 = self.time_emb(t_1).view(-1,1,16,16)
        t_2 = self.time_emb(t_2).view(-1,1,16,16)
        if_1 = self.if_emb(if_1).view(-1,1,16,16)
        if_2 = self.if_emb(if_2).view(-1,1,16,16)
        cls_1 = self.cls_emb(cls_1).view(-1,1,16,16)
        cls_2 = self.cls_emb(cls_2).view(-1,1,16,16)
        x = torch.cat([x, t_1, t_2, if_1, if_2, cls_1, cls_2], dim=1)
        
        x = self.final_block(x)
        return x.view(-1, 1).squeeze(dim=1)
    

'''
Reformulated PatchGAN (NLayerDiscriminator)
Layer N ... N refers to the number of downscaling convs (stride=2)
'''
class Layer5Discriminator(nn.Module):
    def __init__(self, channels, hidden_dim=64, norm_layer=nn.BatchNorm2d):
        super(Layer5Discriminator, self).__init__()

        self.channels = channels
        self.hidden_dim = hidden_dim
        self.norm_layer = norm_layer
        
        self.init_block = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            )
        
        self.main_block = nn.Sequential(
            self._block(hidden_dim,   hidden_dim*2, 4, 2, 1, bias=False),
            self._block(hidden_dim*2, hidden_dim*4, 4, 2, 1, bias=False),
            self._block(hidden_dim*4, hidden_dim*8, 4, 2, 1, bias=False),
            self._block(hidden_dim*8, hidden_dim*8, 4, 2, 1, bias=False),
            self._block(hidden_dim*8, hidden_dim*8, 4, 2, 1, bias=False),
            )
        
        self.final_block = nn.Sequential(
            nn.Conv2d(hidden_dim*8, 1, kernel_size=4, stride=1, padding=0),
            )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            self.norm_layer(out_channels, affine=True),
            nn.LeakyReLU(0.2),
            )
        return block
    
    def forward(self, x):
        x = self.init_block(x)
        x = self.main_block(x)
        x = self.final_block(x)
        return x.view(-1, 1).squeeze(dim=1)
    

'''
Reformulated PatchGAN (NLayerDiscriminator) with included hidden concatenation of t embeddings
Layer N ... N refers to the number of downscaling convs (stride=2)
Output = 1x1
'''
class Layer5Discriminator_t(nn.Module):
    def __init__(self, channels, dim_t, hidden_dim=64, norm_layer=nn.BatchNorm2d):
        super(Layer5Discriminator_t, self).__init__()

        self.channels = channels
        self.dim_t = dim_t
        self.hidden_dim = hidden_dim
        self.norm_layer = norm_layer
        
        self.init_block = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            )
        
        self.main_block = nn.Sequential(
            self._block(hidden_dim,   hidden_dim*2, 4, 2, 1, bias=False),
            self._block(hidden_dim*2, hidden_dim*4, 4, 2, 1, bias=False),
            self._block(hidden_dim*4, hidden_dim*8, 4, 2, 1, bias=False),
            )
        
        self.time_emb = TimeEmbedding(16*16, self.dim_t)
        
        self.final_block = nn.Sequential(
            self._block(hidden_dim*8+1, hidden_dim*8, 4, 2, 1, bias=False),
            self._block(hidden_dim*8, hidden_dim*8, 4, 2, 1, bias=False),
            nn.Conv2d(hidden_dim*8, 1, kernel_size=4, stride=1, padding=0),
            )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            self.norm_layer(out_channels, affine=True),
            nn.LeakyReLU(0.2),
            )
        return block
    
    def forward(self, x, t):
        x = self.init_block(x)
        x = self.main_block(x)
        
        # MLP(EMBED(T)) and concat
        t = self.time_emb(t).view(-1, 1, 16, 16)
        x = torch.cat([x, t], dim=1)
        
        x = self.final_block(x)
        return x.view(-1, 1).squeeze(dim=1)

'''
Reformulated PatchGAN (NLayerDiscriminator) with included hidden concatenation of t embeddings
Layer N ... N refers to the number of downscaling convs (stride=2)
Output = 1x1

Time of img_in and img_gen/ref are encoded with same time_emb layer!
'''
class Layer5Discriminator_img_t(nn.Module):
    def __init__(self, channels, dim_t, hidden_dim=64, norm_layer=nn.BatchNorm2d):
        super(Layer5Discriminator_img_t, self).__init__()

        self.channels = channels
        self.dim_t = dim_t
        self.hidden_dim = hidden_dim
        self.norm_layer = norm_layer
        
        self.init_block = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            )
        
        self.main_block = nn.Sequential(
            self._block(hidden_dim,   hidden_dim*2, 4, 2, 1, bias=False),
            self._block(hidden_dim*2, hidden_dim*4, 4, 2, 1, bias=False),
            self._block(hidden_dim*4, hidden_dim*8, 4, 2, 1, bias=False),
            )
        
        self.time_emb = TimeEmbedding(16*16, self.dim_t)
        
        self.final_block = nn.Sequential(
            self._block(hidden_dim*8+2, hidden_dim*8, 4, 2, 1, bias=False),
            self._block(hidden_dim*8, hidden_dim*8, 4, 2, 1, bias=False),
            nn.Conv2d(hidden_dim*8, 1, kernel_size=4, stride=1, padding=0),
            )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            self.norm_layer(out_channels, affine=True),
            nn.LeakyReLU(0.2),
            )
        return block
    
    def forward(self, x, t_1, t_2):
        x = self.init_block(x)
        x = self.main_block(x)
        
        # MLP(EMBED(T)) and concat
        t_1 = self.time_emb(t_1).view(-1,1,16,16)
        t_2 = self.time_emb(t_2).view(-1,1,16,16)
        x = torch.cat([x, t_1, t_2], dim=1)

        x = self.final_block(x)
        return x.view(-1, 1).squeeze(dim=1)


'''
Reformulated PatchGAN (NLayerDiscriminator) with included hidden concatenation of t embeddings
Layer N ... N refers to the number of downscaling convs (stride=2)
Output = 1x1

Time of img_in and img_gen/ref are encoded with same time_emb layer!
'''
class Layer5Discriminator_img_t_if(nn.Module):
    def __init__(self, channels, dim_t, dim_if, hidden_dim=64, norm_layer=nn.BatchNorm2d):
        super(Layer5Discriminator_img_t_if, self).__init__()

        self.channels = channels
        self.dim_t = dim_t
        self.dim_if = dim_if
        self.hidden_dim = hidden_dim
        self.norm_layer = norm_layer
        
        self.init_block = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            )
        
        self.main_block = nn.Sequential(
            self._block(hidden_dim,   hidden_dim*2, 4, 2, 1, bias=False),
            self._block(hidden_dim*2, hidden_dim*4, 4, 2, 1, bias=False),
            self._block(hidden_dim*4, hidden_dim*8, 4, 2, 1, bias=False),
            )
        
        self.time_emb = TimeEmbedding(16*16, self.dim_t)
        self.if_emb = Embedding(self.dim_if, 16*16)
        
        self.final_block = nn.Sequential(
            self._block(hidden_dim*8+4, hidden_dim*8, 4, 2, 1, bias=False),
            self._block(hidden_dim*8, hidden_dim*8, 4, 2, 1, bias=False),
            nn.Conv2d(hidden_dim*8, 1, kernel_size=4, stride=1, padding=0),
            )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            self.norm_layer(out_channels, affine=True),
            nn.LeakyReLU(0.2),
            )
        return block
    
    def forward(self, x, t_1, t_2, if_1, if_2):
        x = self.init_block(x)
        x = self.main_block(x)
        
        # MLP(EMBED(T)) and concat
        t_1 = self.time_emb(t_1).view(-1,1,16,16)
        t_2 = self.time_emb(t_2).view(-1,1,16,16)
        if_1 = self.if_emb(if_1).view(-1,1,16,16)
        if_2 = self.if_emb(if_2).view(-1,1,16,16)
        x = torch.cat([x, t_1, t_2, if_1, if_2], dim=1)

        x = self.final_block(x)
        return x.view(-1, 1).squeeze(dim=1)


'''
Reformulated PatchGAN (NLayerDiscriminator) with included hidden concatenation of t embeddings
Layer N ... N refers to the number of downscaling convs (stride=2)
Output = 1x1

Time of img_in and img_gen/ref are encoded with same time_emb layer!
'''
class Layer5Discriminator_img_t_cls(nn.Module):
    def __init__(self, channels, dim_t, dim_cls, hidden_dim=64, norm_layer=nn.BatchNorm2d):
        super(Layer5Discriminator_img_t_cls, self).__init__()

        self.channels = channels
        self.dim_t = dim_t
        self.dim_cls = dim_cls
        self.hidden_dim = hidden_dim
        self.norm_layer = norm_layer
        
        self.init_block = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            )
        
        self.main_block = nn.Sequential(
            self._block(hidden_dim,   hidden_dim*2, 4, 2, 1, bias=False),
            self._block(hidden_dim*2, hidden_dim*4, 4, 2, 1, bias=False),
            self._block(hidden_dim*4, hidden_dim*8, 4, 2, 1, bias=False),
            )
        
        self.time_emb = TimeEmbedding(16*16, self.dim_t)
        self.cls_emb = nn.Embedding(self.dim_cls, 16*16)
        
        self.final_block = nn.Sequential(
            self._block(hidden_dim*8+4, hidden_dim*8, 4, 2, 1, bias=False),
            self._block(hidden_dim*8, hidden_dim*8, 4, 2, 1, bias=False),
            nn.Conv2d(hidden_dim*8, 1, kernel_size=4, stride=1, padding=0),
            )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            self.norm_layer(out_channels, affine=True),
            nn.LeakyReLU(0.2),
            )
        return block
    
    def forward(self, x, t_1, t_2, cls_1, cls_2):
        x = self.init_block(x)
        x = self.main_block(x)
        
        # MLP(EMBED(T)) and concat
        t_1 = self.time_emb(t_1).view(-1,1,16,16)
        t_2 = self.time_emb(t_2).view(-1,1,16,16)
        cls_1 = self.cls_emb(cls_1).view(-1,1,16,16)
        cls_2 = self.cls_emb(cls_2).view(-1,1,16,16)
        x = torch.cat([x, t_1, t_2, cls_1, cls_2], dim=1)

        x = self.final_block(x)
        return x.view(-1, 1).squeeze(dim=1)
    

'''
Reformulated PatchGAN (NLayerDiscriminator) with included hidden concatenation of t embeddings
Layer N ... N refers to the number of downscaling convs (stride=2)
Output = 1x1

Time of img_in and img_gen/ref are encoded with same time_emb layer!
'''
class Layer5Discriminator_img_t_if_cls(nn.Module):
    def __init__(self, channels, dim_t, dim_if, dim_cls, hidden_dim=64, norm_layer=nn.BatchNorm2d):
        super(Layer5Discriminator_img_t_if_cls, self).__init__()

        self.channels = channels
        self.dim_t = dim_t
        self.dim_if = dim_if
        self.dim_cls = dim_cls
        self.hidden_dim = hidden_dim
        self.norm_layer = norm_layer
        
        self.init_block = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            )
        
        self.main_block = nn.Sequential(
            self._block(hidden_dim,   hidden_dim*2, 4, 2, 1, bias=False),
            self._block(hidden_dim*2, hidden_dim*4, 4, 2, 1, bias=False),
            self._block(hidden_dim*4, hidden_dim*8, 4, 2, 1, bias=False),
            )
        
        self.time_emb = TimeEmbedding(16*16, self.dim_t)
        self.if_emb = Embedding(self.dim_if, 16*16)
        self.cls_emb = nn.Embedding(self.dim_cls, 16*16)
        
        self.final_block = nn.Sequential(
            self._block(hidden_dim*8+6, hidden_dim*8, 4, 2, 1, bias=False),
            self._block(hidden_dim*8, hidden_dim*8, 4, 2, 1, bias=False),
            nn.Conv2d(hidden_dim*8, 1, kernel_size=4, stride=1, padding=0),
            )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            self.norm_layer(out_channels, affine=True),
            nn.LeakyReLU(0.2),
            )
        return block
    
    def forward(self, x, t_1, t_2, if_1, if_2, cls_1, cls_2):
        x = self.init_block(x)
        x = self.main_block(x)
        
        # MLP(EMBED(T)) and concat
        t_1 = self.time_emb(t_1).view(-1,1,16,16)
        t_2 = self.time_emb(t_2).view(-1,1,16,16)
        if_1 = self.if_emb(if_1).view(-1,1,16,16)
        if_2 = self.if_emb(if_2).view(-1,1,16,16)
        cls_1 = self.cls_emb(cls_1).view(-1,1,16,16)
        cls_2 = self.cls_emb(cls_2).view(-1,1,16,16)
        x = torch.cat([x, t_1, t_2, if_1, if_2, cls_1, cls_2], dim=1)

        x = self.final_block(x)
        return x.view(-1, 1).squeeze(dim=1)
    

'''
code reference: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
'''
class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


'''
code reference: https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/4.%20WGAN-GP/model.py   
'''
class DCGANDiscriminator(nn.Module):
    def __init__(self, channels_img, features_d, img_size):
        super(DCGANDiscriminator, self).__init__()
        self.backend_1 = nn.Sequential(
            # input: N x channels_img x 64 x 64
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            # If img_size is not 64, more blocks are needed to reduce to 4x4 (to have an output of [batch_size,1,1,1] in the end)
        )
        if img_size==64:
            self.backend_2 = nn.Sequential(
                nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
                )
        elif img_size==128:
            self.backend_2 = nn.Sequential(
                self._block(features_d * 8, features_d * 16, 4, 2, 1),
                nn.Conv2d(features_d * 16, 1, kernel_size=4, stride=2, padding=0),
            )
        elif img_size==256:
            self.backend_2 = nn.Sequential(
                self._block(features_d * 8, features_d * 16, 4, 2, 1),
                self._block(features_d * 16, features_d * 32, 4, 2, 1),
                nn.Conv2d(features_d * 32, 1, kernel_size=4, stride=2, padding=0),
                )
        else:
            print('Wrong img_size inserted into discriminator')
            
        self.disc = nn.Sequential(
            self.backend_1,
            self.backend_2)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)    

'''
code reference: Chat-GPT (modified)
'''
class CriticLayerNorm(nn.Module):
    def __init__(self, channels, img_size, hidden_dim=32):
        super(CriticLayerNorm, self).__init__()
        
        self.channels = channels
        self.img_size = img_size
        self.hidden_dim = hidden_dim

        self.main = nn.Sequential(
            # input is (channels) x img_size x img_size
            nn.Conv2d(channels, hidden_dim, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_dim) x img_size/2 x img_size/2
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1, bias=False),
            nn.LayerNorm([hidden_dim * 2, img_size//4, img_size//4]),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_dim*2) x img_size/4 x img_size/4
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, 2, 1, bias=False),
            nn.LayerNorm([hidden_dim * 4, img_size//8, img_size//8]),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_dim*4) x img_size/8 x img_size/8
            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, 4, 2, 1, bias=False),
            nn.LayerNorm([hidden_dim * 8, img_size//16, img_size//16]),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_dim*8) x img_size/16 x img_size/16
            nn.Conv2d(hidden_dim * 8, 1, 4, 1, 0),
        )

    def forward(self, x):
        return self.main(x).squeeze()
    
    
'''
code reference: Chat-GPT (modified)
'''
class CriticInstanceNorm(nn.Module):
    def __init__(self, channels, img_size, hidden_dim=32):
        super(CriticInstanceNorm, self).__init__()
        
        self.channels = channels
        self.img_size = img_size
        self.hidden_dim = hidden_dim

        self.main = nn.Sequential(
            # input is (channels) x img_size x img_size
            nn.Conv2d(channels, hidden_dim, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_dim) x img_size/2 x img_size/2
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_dim*2) x img_size/4 x img_size/4
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_dim*4) x img_size/8 x img_size/8
            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(hidden_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_dim*8) x img_size/16 x img_size/16
            nn.Conv2d(hidden_dim * 8, 1, 4, 1, 0),
        )

    def forward(self, x):
        return self.main(x).squeeze()
