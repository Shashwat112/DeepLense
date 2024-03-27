import torch
import numpy as np
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
import warnings

def mse(x,y):
    return torch.nn.MSELoss()(x,y)

def ssim(x,y):
    m = list()
    for i,j in zip(x,y):
        m.append(structural_similarity(i.numpy(), j.numpy(), channel_axis=0, data_range=1))
    return np.mean(m)

def psnr(x,y):
    m = list()
    for i,j in zip(x,y):
        np.seterr(divide = 'raise')
        try:
            val = peak_signal_noise_ratio(255*i.numpy().astype(np.uint8)[0], 255*j.numpy().astype(np.uint8)[0], data_range=255)
        except:
            continue
        m.append(val)
    return np.mean(m)