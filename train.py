# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 20:59:39 2024

@author: Misha Kaandorp
Publication: 'Incorporating spatial information in deep learning parameter estimation 
with application to the intravoxel incoherent motion model in diffusion-weighted MRI'

Publication: 'A Comparative Study of IVIM-MRI Fitting Techniques in Glioma Grading: 
Conventional, Bayesian, and Voxel-Wise and Spatially-Aware Deep Learning Approaches'
"""

from config import CONFIG

import os
import time
import math
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')

# utils
from utils.dataset_generators import (
    patch_generator_random_uniform,
    patch_generator_random_gaussian,
    patch_generator_structured_uniform,
    patch_generator_structured_gaussian,
    patch_generator_perlin_test,
    generator_perlin_noise
)

from utils.util_functions import Descale_params, Scale_params, IVIM_model
from utils.loss_functions import calculate_losses
from utils.models import TransformerNet


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#################################################
# Load configuration
#################################################

cfg = CONFIG

model_name = cfg["model_name"]

training_cfg = cfg["training"]
dataset_cfg = cfg["dataset"]
ivim_cfg = cfg["ivim"]
model_cfg = cfg["models"][model_name]
path_cfg = cfg["paths"]

device = torch.device("cuda:0" if cfg["device"] == "gpu" else "cpu")

if cfg["device"] == "gpu":
    torch.backends.cudnn.benchmark = True

print(f"Using device: {device}")
print("Experiment configuration:")
print(CONFIG)

#################################################
# Model parameters
#################################################

hidden_channels = model_cfg["hidden_channels"]
num_layers = model_cfg["num_layers"]
kernel_NATTEN = model_cfg["kernel_NATTEN"]
norm_first = model_cfg["norm_first"]
attention = model_cfg["attention"]
out_channels = model_cfg["out_channels"]

#################################################
# Training parameters
#################################################

training_generator = training_cfg["training_generator"]
training_strategy = training_cfg["training_strategy"]
learning_rate = training_cfg["learning_rate"]
n_epochs = training_cfg["n_epochs"]
batches_per_epoch = training_cfg["batches_per_epoch"]
batch_size = training_cfg["batch_size"]
patch_size = training_cfg["patch_size"]


#################################################
# Paths
#################################################

resume_weight = path_cfg["resume_weight"]
output_dir = path_cfg["output_dir"]

#################################################
# Loss
#################################################

criterion = torch.nn.MSELoss()

#################################################
# IVIM experiment parameters
#################################################

snr = ivim_cfg["snr"]
bval = ivim_cfg["bvalues"]
bounds = np.array(ivim_cfg["bounds"])

b_torch = torch.Tensor(bval).to(device)[None, :, None, None]

#################################################
# Dataset generators
#################################################

generators = {
    'patch_generator_random_uniform': patch_generator_random_uniform,
    'patch_generator_random_gaussian': patch_generator_random_gaussian,
    'patch_generator_structured_uniform': patch_generator_structured_uniform,
    'patch_generator_structured_gaussian': patch_generator_structured_gaussian,
    'patch_generator_perlin_test': patch_generator_perlin_test,
    'generator_perlin_noise': generator_perlin_noise
}

gen_fn = generators.get(training_generator)

if gen_fn is None:
    raise ValueError(f"Unknown dataset generator: {training_generator}")

gen = gen_fn(batch_size, patch_size, device, b_torch, bounds, snr)

signal_noise_val, params_scaled_val, params_val = next(gen)

#################################################
# Network
#################################################

if model_name in ["SA-17", "NATTEN-17"]:

    net = TransformerNet(
        bval,
        out_channels,
        kernel_NATTEN=kernel_NATTEN,
        norm_first=norm_first,
        attention=attention,
        num_layers_Transf=num_layers,
        dim_feedforward=hidden_channels,
        nhead=8
    ).to(device)

#################################################
# Optimizer
#################################################

opt = torch.optim.Adam(net.parameters(), lr=learning_rate)

#################################################
# Output directory
#################################################

output_dir = Path(output_dir) / 'training' / model_name
output_dir.mkdir(parents=True, exist_ok=True)

#################################################
# Resume training if needed
#################################################

epoch = 0

if len(resume_weight) > 0:

    print(f"Loading model {resume_weight}")
    net.load_state_dict(torch.load(resume_weight, map_location=device))
    print("Model loaded!")

#################################################
# Training
#################################################

print("#####  TRAIN   #####")

start_time_epoch = time.time()

middle_ps = math.floor(patch_size / 2)

for epoch in range(epoch, n_epochs):

    for batch in range(batches_per_epoch):

        opt.zero_grad()

        net.train()

        if epoch < 2 and batch % 20 == 0:
            print(f"Train epoch {epoch} batch {batch}")

        signal_noise_train, params_scaled_train, params_train = next(gen)

        #################################################
        # Forward pass
        #################################################

        out_network_train = net(signal_noise_train.to(device))

        train_params_descaled = abs(Descale_params(out_network_train, bounds))

        out_network_train_rescaled = Scale_params(train_params_descaled, bounds)

        signal_pred_train = IVIM_model(
            train_params_descaled[:, 0][:, None], # S0
            train_params_descaled[:, 1][:, None], # D
            train_params_descaled[:, 2][:, None], # f
            train_params_descaled[:, 3][:, None], # D*
            b_torch
        )

        #################################################
        # Extract center voxel
        #################################################

        if model_name in ["SA-17", "NATTEN-17"]:

            signal_noise_train = signal_noise_train[:, :, middle_ps, middle_ps]
            signal_pred_train = signal_pred_train[:, :, middle_ps, middle_ps]

            params_scaled_train = params_scaled_train[:, :, middle_ps, middle_ps]
            out_network_train_rescaled = out_network_train_rescaled[:, :, middle_ps, middle_ps]

        #################################################
        # Loss
        #################################################

        losses_train = calculate_losses(
            criterion,
            out_network_train_rescaled,
            params_scaled_train,
            signal_noise_train,
            signal_pred_train
        )

        if training_strategy == 'supervised':
            loss_train = losses_train['parameters'].float()

        elif training_strategy == 'self-supervised':
            loss_train = losses_train['signals'].float()

        #################################################
        # Backprop
        #################################################

        loss_train.backward()
        opt.step()

    #################################################
    # Epoch log
    #################################################

    print(f'epoch {epoch} loss: {loss_train} time: {time.time() - start_time_epoch}')

    #################################################
    # Save checkpoints
    #################################################

    if epoch % 10 == 0:

        print(f"Saving model at epoch {epoch}")

        torch.save(net.state_dict(), output_dir / f'model_epoch_{epoch}.pt')

        torch.save(net.state_dict(), output_dir / f'final_model.pt')