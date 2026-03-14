# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 20:59:39 2024

@author: Misha Kaandorp
Publication: 'Incorporating spatial information in deep learning parameter estimation 
with application to the intravoxel incoherent motion model in diffusion-weighted MRI'

Publication: 'A Comparative Study of IVIM-MRI Fitting Techniques in Glioma Grading: 
Conventional, Bayesian, and Voxel-Wise and Spatially-Aware Deep Learning Approaches'
"""

import torch
import math

def calculate_losses(criterion, out_network, params_scaled, signal, signal_pred):
    losses = {}
    losses['signals'] = criterion(signal, signal_pred)
    for index,name in zip([0,1,2,3], ['S0', 'D', 'F', 'D*']):
        losses[name] = criterion(out_network[:,index].float(), params_scaled[:,index].float())
    losses['parameters'] = criterion(out_network.float(), params_scaled.float())
    return losses
