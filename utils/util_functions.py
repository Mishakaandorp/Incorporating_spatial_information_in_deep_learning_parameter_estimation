# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 20:42:01 2024

@author: Misha Kaandorp
Publication: 'Incorporating spatial information in deep learning parameter estimation 
with application to the intravoxel incoherent motion model in diffusion-weighted MRI'

Publication: 'A Comparative Study of IVIM-MRI Fitting Techniques in Glioma Grading: 
Conventional, Bayesian, and Voxel-Wise and Spatially-Aware Deep Learning Approaches'

"""
import torch
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt


def IVIM_model(S0, Dt, Fp, Dp, bvalues):
    return (S0 * (Fp * torch.exp(-bvalues * Dp) + (1 - Fp) * torch.exp(-bvalues * Dt)))

def descale_params(paramnorm, Lower_bound, Upper_bound):
    a1 = 0
    b1 = 1
    return ((((paramnorm - a1) / (b1 - a1)) ) * (Upper_bound - Lower_bound)) + Lower_bound

def scaling(param, Lower_bound, Upper_bound):
    a1 = 0
    b1 = 1
    return (b1 - a1) * ((param - Lower_bound) / (Upper_bound - Lower_bound)) + a1

def Descale_params(params, bounds):
    S0_descaled = descale_params(params[:,0],bounds[0,0], bounds[1,0])
    D_descaled = descale_params(params[:,1],bounds[0,1], bounds[1,1])
    F_descaled = descale_params(params[:,2],bounds[0,2], bounds[1,2])
    Dp_descaled = descale_params(params[:,3],bounds[0,3], bounds[1,3])
    params_descaled = torch.cat((S0_descaled[:,None], D_descaled[:,None], F_descaled[:,None], Dp_descaled[:,None]), axis=1)
    return params_descaled

def Scale_params(params, bounds):
    S0_scaled = scaling(params[:,0],bounds[0,0], bounds[1,0])
    D_scaled = scaling(params[:,1],bounds[0,1], bounds[1,1])
    F_scaled = scaling(params[:,2],bounds[0,2], bounds[1,2])
    Dp_scaled = scaling(params[:,3],bounds[0,3], bounds[1,3])
    params_scaled = torch.cat((S0_scaled[:,None], D_scaled[:,None], F_scaled[:,None], Dp_scaled[:,None]), axis=1)
    return params_scaled

def signal_params(params, snr, b_torch, device):
    # Create signal with IVIM model
    signal = IVIM_model(params[:,0][:,None], params[:,1][:,None], params[:,2][:,None], params[:,3][:,None], b_torch)            
    # Add complex-valued noise
    signal_noise = signal + 1/snr * (torch.randn(signal.shape, device=device) + 1j*torch.randn(signal.shape, device=device))
    return signal_noise

def extract_image_patches(x, kernel, stride=1, dilation=1):
    b,c,h,w = x.shape
    h2 = math.ceil(h / stride)
    w2 = math.ceil(w / stride)
    pad_row = (h2 - 1) * stride + (kernel - 1) * dilation + 1 - h
    pad_col = (w2 - 1) * stride + (kernel - 1) * dilation + 1 - w
    x = F.pad(x, (pad_row//2, pad_row - pad_row//2, pad_col//2, pad_col - pad_col//2))
   
    # Extract patches
    patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
    patches = patches.permute(0,4,5,1,2,3).contiguous()
    return patches
