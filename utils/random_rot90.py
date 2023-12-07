import torch
# import numpy as np
import random

class RandomRot90(object):
    """Randomly rotate 0/90/180/270
    Expects by default image in 'channel first' order [C,W,H]
    Args:
        dims (list or tuple): axis to rotate Default [1,2]
    """
    def __init__(self, dims=[1,2]):
        self.dims=dims

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Rotated image
        """
        # how often to rotate
        k = random.randint(1,4)
        
        return torch.rot90(img,k,self.dims)