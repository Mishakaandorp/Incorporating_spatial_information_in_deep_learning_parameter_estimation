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
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import pickle
import math
from pathlib import Path

from utils.util_functions import (
    Descale_params,
    Scale_params,
    IVIM_model,
    extract_image_patches
)

from utils.models import TransformerNet

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
network = model_name

patch_size = training_cfg["patch_size"]

#################################################
# Dataset parameters
#################################################

GT_and_error_maps_flag = dataset_cfg["GT_and_error_maps_flag"]

#################################################
# IVIM parameters
#################################################

snr = ivim_cfg["snr"]
bval = ivim_cfg["bvalues"]
bounds = np.array(ivim_cfg["bounds"])

b_torch = torch.Tensor(bval).to(device)[None, :, None, None]

#################################################
# Paths
#################################################

run_dir = Path(path_cfg["output_dir"])
dataset_name = "fractal_noise"
evaluated_network = "final_model"

#################################################
# Load dataset
#################################################

dataset_file = run_dir  / "datasets" / "fractal_noise" / f"{dataset_name}.pkl"

with open(dataset_file, "rb") as f:
    signal_noise_val, params_scaled_val, params_val = pickle.load(f)

signal_noise_val = signal_noise_val.to(device)

#################################################
# Build network
#################################################

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
# Load weights
#################################################
model_path = run_dir / "training" / network / f"{evaluated_network}.pt"
net.load_state_dict(torch.load(model_path, map_location=device))

print(f"Loaded model: {model_path}")

net.eval()

#################################################
# Initialize outputs
#################################################

out_network_descaled_all = torch.zeros(
    signal_noise_val.size(0),
    out_channels,
    signal_noise_val.size(2),
    signal_noise_val.size(3)
)

signal_pred_all = torch.zeros(signal_noise_val.size())

#################################################
# Run inference
#################################################

print("#####  Inference   #####")


for map_ in range(signal_noise_val.size(0)):
    print(f"Processing slice {map_}")
    signal_noise_val_map = signal_noise_val[map_][None, :]
    if network == "SA-17":
        with torch.no_grad():
            patches = extract_image_patches(signal_noise_val_map, patch_size)
            image_brain_slice_per = patches.permute(0,4,5,3,2,1).contiguous()
            s,x,y,b_,px,py = image_brain_slice_per.shape
            signal_noise_val_new = image_brain_slice_per.reshape(s*x*y, b_, px, py)
            batch_stepsize = 1024   # can be lower depending on RAM
            batch_steps = np.arange(0, signal_noise_val_new.shape[0]+1, batch_stepsize)

            if batch_steps[-1] != signal_noise_val_new.shape[0]:
                batch_steps = np.append(batch_steps, signal_noise_val_new.shape[0])
            
            steps_all = []
            for step in range(len(batch_steps)-1):
                batch_out = net(
                    signal_noise_val_new[
                        batch_steps[step]:batch_steps[step+1]
                    ].to(device)
                )
                steps_all.append(batch_out)

            out_network_val_net = torch.cat(steps_all, dim=0)
            middle_kernel = math.floor(patch_size/2)
            out_network_val_all = out_network_val_net.reshape(s,x,y,out_channels,px,py)
            out_network_val_middle = out_network_val_all[:,:,:,:,middle_kernel,middle_kernel]
            out_network_val = out_network_val_middle.permute(0,3,1,2)

    else:
        with torch.no_grad():
            out_network_val = net(signal_noise_val_map.to(device))

    #################################################
    # Convert outputs
    #################################################

    out_network_descaled = abs(Descale_params(out_network_val, bounds))
    out_network_val_rescaled = Scale_params(out_network_descaled, bounds)
    signal_pred = IVIM_model(
        out_network_descaled[:,0][:,None],
        out_network_descaled[:,1][:,None],
        out_network_descaled[:,2][:,None],
        out_network_descaled[:,3][:,None],
        b_torch
    )
    signal_pred_all[map_] = signal_pred
    out_network_descaled_all[map_] = out_network_descaled

#################################################
# Save evaluation figures
#################################################

def save_ivim_figures(
    out_network_descaled_all,
    signal_pred_all,
    signal_noise_val,
    params_val,
    eval_dir,
    map_,
    colormap="jet",
    GT_and_error_maps=True  # <-- only generate GT and AE figures if True
):
    """
    Saves IVIM predicted figures for one slice.
    Optionally also saves ground truth and absolute error maps.

    Parameters
    ----------
    out_network_descaled_all : torch.Tensor or np.ndarray
        Predicted IVIM parameter maps, shape (N, 4, H, W)
    signal_pred_all : torch.Tensor or np.ndarray
        Predicted signal maps
    signal_noise_val : torch.Tensor or np.ndarray
        Measured/noisy signal maps
    params_val : torch.Tensor or np.ndarray
        Ground truth IVIM parameter maps
    eval_dir : Path
        Root directory to save figures
    map_ : int
        Slice index
    colormap : str
        Matplotlib colormap
    GT_and_error_maps : bool
        If False, skip saving GT and absolute error figures
    """

    print(f"Making predicted figures for slice {map_}")

    slice_dir = eval_dir / str(map_)
    slice_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Predicted parameter maps
    # -----------------------------
    IVIM_param = {}
    IVIM_param["S0"] = out_network_descaled_all[:,0]
    IVIM_param["D"] = out_network_descaled_all[:,1]
    IVIM_param["F"] = out_network_descaled_all[:,2] * 100 # in percentage
    IVIM_param["Dp"] = out_network_descaled_all[:,3]

    # Save predicted maps
    pred_limits = {
        "S0": (0, 1),
        "D": (0, 0.003),
        "F": (0, 50),
        "Dp": (0, 0.05),
    }

    for name_param, (vmin, vmax) in pred_limits.items():
        plt.figure()
        plt.axis("off")
        img = IVIM_param[name_param][map_]
        if torch.is_tensor(img):
            img = img.cpu().detach().numpy()
        plt.imshow(img, cmap=colormap, vmin=vmin, vmax=vmax)
        plt.savefig(slice_dir / f"{name_param}.png", dpi=600, transparent=True, bbox_inches="tight")
        plt.close()

    # -----------------------------
    # RMSE map
    # -----------------------------
    RMSE_diff = np.sqrt(
        (
            abs(signal_pred_all[map_].cpu().numpy() - signal_noise_val[map_].cpu().numpy()) ** 2
        ).mean(axis=0)
    )
    plt.figure()
    plt.axis("off")
    plt.imshow(RMSE_diff, cmap=colormap, vmin=0, vmax=0.02)
    plt.savefig(slice_dir / "RMSE.png", dpi=600, transparent=True, bbox_inches="tight")
    plt.close()

    # -----------------------------
    # Ground truth and absolute error maps
    # -----------------------------
    if GT_and_error_maps:

        print(f"Making GT and absolute error figures for slice {map_}")

        # Ground truth maps
        GT_maps = {
            "S0": (params_val[map_,0], 0, 1),
            "D":  (params_val[map_,1], 0, 0.003),
            "F":  (params_val[map_,2]*100, 0, 50),
            "Dp": (params_val[map_,3], 0, 0.05),
        }

        for name_param, (img, vmin, vmax) in GT_maps.items():
            plt.figure()
            plt.axis("off")
            if torch.is_tensor(img):
                img = img.cpu().detach().numpy()
            plt.imshow(img, cmap=colormap, vmin=vmin, vmax=vmax)
            plt.savefig(slice_dir / f"{name_param}_GT.png", dpi=600, transparent=True, bbox_inches="tight")
            plt.close()

        # Absolute error maps
        pred = out_network_descaled_all.cpu().detach().numpy() if torch.is_tensor(out_network_descaled_all) else out_network_descaled_all
        gt = params_val.cpu().detach().numpy() if torch.is_tensor(params_val) else params_val
        absolute_parameter_errors = np.abs(gt - pred)

        AE_maps = {
            "S0": (absolute_parameter_errors[:,0], 0, 1),
            "D":  (absolute_parameter_errors[:,1], 0, 0.003),
            "F":  (absolute_parameter_errors[:,2]*100, 0, 50),
            "Dp": (absolute_parameter_errors[:,3], 0, 0.05),
        }

        for name_param, (img_stack, vmin, vmax) in AE_maps.items():
            img = img_stack[map_]
            plt.figure()
            plt.axis("off")
            plt.imshow(img, cmap=colormap, vmin=vmin, vmax=vmax)
            plt.savefig(slice_dir / f"{name_param}_AE.png", dpi=600, transparent=True, bbox_inches="tight")
            plt.close()


eval_dir = run_dir / "evaluate" / network / dataset_name / evaluated_network
print(f"evaluation dir: {eval_dir}")

GT_and_error_maps_flag = True

print(f'##################################################')
print(f'#####   Note!!! GT and error maps can only be made with simulations GT, not with in vivo data    ######')
print(f'#####   Currently: GT_and_error_maps_flag = {GT_and_error_maps_flag}    ######')
print(f'##################################################')

for map_ in range(signal_noise_val.size(0)):
    save_ivim_figures(
        out_network_descaled_all=out_network_descaled_all,
        signal_pred_all=signal_pred_all,
        signal_noise_val=signal_noise_val,
        params_val=params_val,
        eval_dir=eval_dir,
        map_=map_,
        colormap="jet",
        GT_and_error_maps=GT_and_error_maps_flag
    )

    