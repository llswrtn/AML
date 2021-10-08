# Alternative functions to transform image for pytorch torchvision faster r-cnn adapted from official repo:  https://github.com/pytorch/vision/tree/main/references/detection

import numpy as np
import pandas as pd
import torch
from typing import List, Tuple, Dict, Optional, Any


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor(torch.nn.Module):
    def forward(self, image: torch.Tensor,
                target: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        image = to_tensor(image)
        return image, target


def get_transform(train):
    transforms = []
    transforms.append(ToTensor())
    #no data augmentation for now, maybe implement later
    #if train:
        #transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)


    
def to_tensor(pic):

    default_float_dtype = torch.get_default_dtype()

    if isinstance(pic, np.ndarray):
        # handle numpy array
        #print("# handle numpy array in utils.to_tensor")
        if pic.ndim == 2:
            pic = pic[:, :, None]

        img = torch.from_numpy(pic.transpose((2, 0, 1))).contiguous()
        return img 


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)
