import torch
import numpy as np
import random

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = random.randint(0,h)
            x = random.randint(0,w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
    
    
class Shadowout(object):
    """Randomly mask out one or more non-quadratic patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        min_length (int): Min length (in pixels) of each patch.
        max_length (int): Max length (in pixels) of each patch.
    """
    def __init__(self, n_holes, min_length, max_length):
        self.n_holes = n_holes
        self.min_length = min_length
        self.max_length = max_length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = random.randint(0,h)
            x = random.randint(0,w)
            
            x_length = random.randint(self.min_length, self.max_length)
            y_length = random.randint(self.min_length, self.max_length)

            y1 = np.clip(y - y_length // 2, 0, h)
            y2 = np.clip(y + y_length // 2, 0, h)
            x1 = np.clip(x - x_length // 2, 0, w)
            x2 = np.clip(x + x_length // 2, 0, w)

            mask[y1: y2, x1: x2] = random.random()

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask.to(img)

        return img