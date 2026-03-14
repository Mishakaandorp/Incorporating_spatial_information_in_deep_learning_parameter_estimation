# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 09:00:48 2023

@author: Misha Kaandorp
Publication: 'Incorporating spatial information in deep learning parameter estimation 
with application to the intravoxel incoherent motion model in diffusion-weighted MRI'

Publication: 'A Comparative Study of IVIM-MRI Fitting Techniques in Glioma Grading: 
Conventional, Bayesian, and Voxel-Wise and Spatially-Aware Deep Learning Approaches'
"""
import torch
import math
import matplotlib.pyplot as plt
import numpy as np

from utils.transformer_NATTEN import TransformerEncoderLayerNATTEN

# Self-attention transformer
class TransformerNet(torch.nn.Module):
    """
    A transformer net, based on: "Dosovitskiy et al.,
    """

    def __init__(
        self, b, out_channels,  kernel_NATTEN, norm_first, attention, num_layers_Transf, dim_feedforward: int, nhead: int, dropout: float = 0.0, qkv_bias: bool = False
    ) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            qkv_bias: apply bias term for the qkv linear layer

        """
        super().__init__()
        # Define attention
        self.attention = attention
        # Linear layer
        self.linearIN= torch.nn.Linear(len(b),dim_feedforward)
        # Transformer encoder block
        self.transformer_encoder = torch.nn.TransformerEncoder(TransformerEncoderLayerNATTEN(dim_feedforward, attention, kernel_NATTEN, nhead, dim_feedforward, dropout= dropout, activation="relu", norm_first=norm_first), num_layers = num_layers_Transf)
        # Output layer for the defined output nodes (model parameters)
        self.linearOUT= torch.nn.Linear(dim_feedforward,out_channels)
    
    def forward(self, x):
        # Either neighborhood attention or self-attention
        # input x is in (batchsize, b, x, y). Here x,y is the patch size

        if self.attention == 'NATTEN':
            # permute to (batchsize, x, y, b)
            xT = x.permute(0,2,3,1)
            # linear layer to units transformer. So, (batchsize, x, y, u)
            x_l1 = self.linearIN(xT)
            # to transformerencoder
            x_TE = self.transformer_encoder(x_l1)
            # transform output to 4 output params (S0, D, f, D*)
            # output will be (batchsize, x, y, 4)
            x_out = self.linearOUT(x_TE)
            # permute to (batchsize, 4, x, y)
            x_out = x_out.permute(0,3,1,2)
        elif self.attention == 'SELF':
            # permute to (x, y, batchsize, b)
            xT = x.permute(2,3,0,1)
            # reshape to (x*y, batchsize, b)
            x_view = xT.view(xT.size(0)*xT.size(1),xT.size(2),xT.size(3))
            # linear layer to units transformer. So, (x*y, batchsize, u)
            x_l1 = self.linearIN(x_view)
            # to transformerencoder
            x_TE = self.transformer_encoder(x_l1)
            # transform output to 4 output params (S0, D, f, D*)
            # output will be (x*y, batchsize, 4)
            x_out = self.linearOUT(x_TE)
            # reshape output to (x, y, batchsize, 4)
            x_out = x_out.view(int(x_out.size(0)**0.5),int(x_out.size(0)**0.5),x_out.size(1),x_out.size(2))
            # permute to (batchsize, 4, x, y)
            x_out = x_out.permute(2,3,0,1)
        return x_out