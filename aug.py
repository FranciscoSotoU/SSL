import torch
from scipy.interpolate import interp1d
import numpy as np
import random
from torchvision import transforms
def add_noise(tensor):
    std = 0.1
    mean = 0
    noise = torch.randn(input.size()) * std + mean
    tensor = tensor + noise
    return tensor



def time_warp(tensor):
    warp_factor = 0.7
    num_samples, sequence_length = tensor.size()
    warped_tensor = torch.zeros_like(tensor)


    t = torch.linspace(0, 1, sequence_length)
    t_warped = warp_factor * t + (1 - warp_factor) * t.mean()

    for i in range(num_samples):
        f = interp1d(t, tensor[i, :])
        warped_tensor[i, :] = torch.from_numpy(f(t_warped))

    return warped_tensor

def time_flip(tensor):
    time = tensor[0]
    mag = tensor[1]
    time_flipped = torch.flip(time,dims=[0])

    return torch.stack([time_flipped,mag])

def mag_flip(tensor):
    time = tensor[0]
    mag = tensor[1]
    mag_flipped = -mag

    return torch.stack([time,mag_flipped])

def zero_masking(tensor):
    factor = 0.3
    largo = len(tensor[1])
    mask_len = int(largo * factor)
    mask = np.ones(largo,dtype=bool)
    start = int(random.randint(0,int(largo-mask_len)))
    end = int(start + mask_len)
    if end>largo:
        end = largo
    mask[ start:end  ] = False
    mask = torch.tensor(mask.astype(int))
    values = tensor[1] * mask
    times = tensor[0]

    return torch.stack([times,values])



    
def augment_pool():
    augs = [(zero_masking),
            (mag_flip),
            (time_flip),
            (time_warp),
            (add_noise)]
    return augs
