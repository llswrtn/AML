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
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    This function does not support torchscript.

    See :class:`~torchvision.transforms.ToTensor` for more details.

    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """


    default_float_dtype = torch.get_default_dtype()

    if isinstance(pic, np.ndarray):
        # handle numpy array
        #print("# handle numpy array in utils.to_tensor")
        if pic.ndim == 2:
            pic = pic[:, :, None]

        img = torch.from_numpy(pic.transpose((2, 0, 1))).contiguous()
        return img 
        '''
        # backward compatibility
        
        return img
        
        if isinstance(img, torch.ByteTensor):
            print("#backward compatibility in utils.to_tensor if")
            return img.to(dtype=default_float_dtype).div(255)
        else:
            print("#backward compatibility in utils.to_tensor else")
            return img
        '''
        
    '''
    if accimage is not None and isinstance(pic, accimage.Image):
        print("accimage")
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return torch.from_numpy(nppic).to(dtype=default_float_dtype)

    # handle PIL Image
    mode_to_nptype = {'I': np.int32, 'I;16': np.int16, 'F': np.float32}
    img = torch.from_numpy(
        np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True)
    )

    if pic.mode == '1':
        img = 255 * img
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    # put it from HWC to CHW format
    img = img.permute((2, 0, 1)).contiguous()
    if isinstance(img, torch.ByteTensor):
        print("torch byte tensor")
        return img.to(dtype=default_float_dtype).div(255)
    else:
        return img
    '''
