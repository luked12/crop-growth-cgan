import torch
import torch.nn as nn    


class ResNet18DecoderCbnNoPool(nn.Module):

    n_blocks = [1,2,2,2,2]
    base_channels = 64

    def __init__(self, n_aux, cat_channels=0):
        super().__init__()
        self.cat_channels = cat_channels
        
        self.conv_out = nn.Conv2d(self.base_channels, 3, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')#, align_corners=True)
        
        self.layer_init = ResLayerCbn(in_channels=self.base_channels*8+self.cat_channels, out_channels=self.base_channels*8, n_blocks=self.n_blocks[0], n_aux=n_aux)
        self.layer_1 = ResLayerCbn(in_channels=self.base_channels*8, out_channels=self.base_channels*4, n_blocks=self.n_blocks[1], n_aux=n_aux)
        self.layer_2 = ResLayerCbn(in_channels=self.base_channels*4, out_channels=self.base_channels*2, n_blocks=self.n_blocks[2], n_aux=n_aux)
        self.layer_3 = ResLayerCbn(in_channels=self.base_channels*2, out_channels=self.base_channels, n_blocks=self.n_blocks[3], n_aux=n_aux)
        self.layer_4 = ResLayerCbn(in_channels=self.base_channels, out_channels=self.base_channels, n_blocks=self.n_blocks[4], n_aux=n_aux)
        
    def forward(self, x, aux):
        x = self.layer_init(x,aux)
        x = self.upsample(x)
        x = self.layer_1(x, aux)
        x = self.upsample(x)
        x = self.layer_2(x, aux)
        x = self.upsample(x)
        x = self.layer_3(x, aux)
        x = self.upsample(x)
        x = self.layer_4(x, aux)
        x = self.conv_out(x)
        
        return x
 
    
class ResNet18DecoderCbn(nn.Module):

    n_blocks = [1,2,2,2,2,2,2,2,2]
    base_channels = 64

    def __init__(self, n_aux, cat_channels=0):
        super().__init__()
        self.cat_channels = cat_channels
        
        self.conv_out = nn.Conv2d(self.base_channels, 3, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')#, align_corners=True)
        
        self.layer_init = ResLayerCbn(in_channels=self.base_channels*8+self.cat_channels, out_channels=self.base_channels*8, n_blocks=self.n_blocks[0], n_aux=n_aux)
        self.layer_1 = ResLayerCbn(in_channels=self.base_channels*8, out_channels=self.base_channels*8, n_blocks=self.n_blocks[1], n_aux=n_aux)
        self.layer_2 = ResLayerCbn(in_channels=self.base_channels*8, out_channels=self.base_channels*4, n_blocks=self.n_blocks[2], n_aux=n_aux)
        self.layer_3 = ResLayerCbn(in_channels=self.base_channels*4, out_channels=self.base_channels*4, n_blocks=self.n_blocks[3], n_aux=n_aux)
        self.layer_4 = ResLayerCbn(in_channels=self.base_channels*4, out_channels=self.base_channels*2, n_blocks=self.n_blocks[4], n_aux=n_aux)
        self.layer_5 = ResLayerCbn(in_channels=self.base_channels*2, out_channels=self.base_channels*2, n_blocks=self.n_blocks[5], n_aux=n_aux)
        self.layer_6 = ResLayerCbn(in_channels=self.base_channels*2, out_channels=self.base_channels, n_blocks=self.n_blocks[6], n_aux=n_aux)
        self.layer_7 = ResLayerCbn(in_channels=self.base_channels, out_channels=self.base_channels, n_blocks=self.n_blocks[7], n_aux=n_aux)
        self.layer_8 = ResLayerCbn(in_channels=self.base_channels, out_channels=self.base_channels, n_blocks=self.n_blocks[8], n_aux=n_aux)

        
    def forward(self, x, aux):
        x = self.layer_init(x,aux)
        x = self.upsample(x)
        x = self.layer_1(x, aux)
        x = self.upsample(x)
        x = self.layer_2(x, aux)
        x = self.upsample(x)
        x = self.layer_3(x, aux)
        x = self.upsample(x)
        x = self.layer_4(x, aux)
        x = self.upsample(x)
        x = self.layer_5(x, aux)
        x = self.upsample(x)
        x = self.layer_6(x, aux)
        x = self.upsample(x)
        x = self.layer_7(x, aux)
        x = self.upsample(x)
        x = self.layer_8(x, aux)
        x = self.conv_out(x)
        
        return x    


class ResLayerCbn(nn.Module):

    def __init__(self, in_channels, out_channels, n_blocks, n_aux, downsample=False):

        super().__init__()

        self.blocks = nn.ModuleList()
        self.blocks.append(ResBlockCbn(in_channels, out_channels, n_aux, downsample=downsample))
        
        for _ in range(n_blocks-1):
            self.blocks.append(ResBlockCbn(out_channels, out_channels, n_aux))
        
    def forward(self, x, aux):

        for block in self.blocks:
            x = block(x, aux)
        
        return x

class ResBlockCbn(nn.Module):
    
    def __init__(self, in_channels, out_channels, n_aux, downsample=False):

        super().__init__()

        stride = 2 if downsample else 1
        self.identity_layer = (in_channels != out_channels)
        
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batchnorm_1 = ConditionalBatchNorm2d(out_channels, n_aux)
        
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.batchnorm_2 = ConditionalBatchNorm2d(out_channels, n_aux)
        
        self.conv_i = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1) if self.identity_layer else None
        self.batchnorm_i = ConditionalBatchNorm2d(out_channels, n_aux) if self.identity_layer else None

        self.relu = nn.ReLU()
        
    def forward(self, x, aux):

        identity = x.clone()

        if self.identity_layer:
            identity = self.batchnorm_i(self.conv_i(identity), aux)
        
        x = self.relu(self.batchnorm_1(self.conv_1(x), aux))
        x = self.batchnorm_2(self.conv_2(x), aux)
        
        x += identity
        x = self.relu(x)
                
        return x

class ConditionalBatchNorm2d(nn.Module):
  
    def __init__(self, n_features, n_aux):

        super().__init__()
        
        self.n_features = n_features
        self.bn = nn.BatchNorm2d(n_features, affine=False)
        self.linear = nn.Linear(n_aux, n_features * 2)

    def forward(self, x, aux):

        out = self.bn(x)
        gamma, beta = self.linear(aux).chunk(2, 1)
        out = gamma.view(-1, self.n_features, 1, 1) * out + beta.view(-1, self.n_features, 1, 1)
        
        return out




