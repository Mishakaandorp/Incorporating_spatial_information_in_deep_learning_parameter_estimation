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
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import os
from pathlib import Path

from utils.dataset_generators import generator_perlin_noise_patch
from utils.util_functions import IVIM_model
from config import CONFIG  # <-- everything comes from config

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
matplotlib.use('Agg')


def save_param_figure(params, save_dir, sample_idx=0):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    p = params[sample_idx].cpu().numpy()  # shape (4, H, W)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    titles = ["S0", "D", "f", "D*"]

    for i, ax in enumerate(axes.flat):
        im = ax.imshow(p[i], cmap="jet")
        ax.set_title(titles[i])
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path = Path(save_dir) / f"params_sample_{sample_idx}.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved figure: {save_path}")


def main():

    ########################################
    # Load generator parameters from config
    ########################################
    ds_config = CONFIG["dataset"]
    generator_name = ds_config["generator"]
    device_type = ds_config["device"]
    fractal_params = ds_config["fractal_noise"]
    ivim_params = CONFIG["ivim"]

    ########################################
    # Device setup
    ########################################
    device_type = CONFIG["dataset"].get("device", CONFIG["device"])
    if device_type == 'gpu':
        torch.backends.cudnn.benchmark = True
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = torch.device("cuda:0")
        print(f'Device: {torch.cuda.get_device_name(device.index)}')
    else:
        device = torch.device('cpu')
        torch.set_num_threads(1)
        print("Using CPU")

    ########################################
    # Load params
    ########################################

    batch_size = fractal_params["batch_size"]
    image_shape = tuple(fractal_params["image_shape"])
    res = fractal_params["res"]
    octaves = fractal_params["octaves"]

    snr = ivim_params["snr"]
    bounds = np.array(ivim_params["bounds"])
    bval = ivim_params["bvalues"]
    b_torch = torch.Tensor(bval).to(device)[None, :, None, None]

    output_dir = Path(CONFIG["paths"]["output_dir"]) / "datasets" / "fractal_noise"
    output_dir.mkdir(parents=True, exist_ok=True)

    ########################################
    # Generate dataset
    ########################################

    print("#####  Generate fractal-noise dataset   #####")

    if generator_name == "generator_perlin_noise_patch":
        signal_noise, params_scaled, params = next(
            generator_perlin_noise_patch(
                batch_size=batch_size,
                image_shape=image_shape,
                device=device,
                b_torch=b_torch,
                bounds=bounds,
                snr=snr,
                res=res,
                octaves=octaves
            )
        )
    else:
        raise ValueError(f"Unknown dataset generator: {generator_name}")

    ########################################
    # Save dataset and example figure
    ########################################
    save_param_figure(params, save_dir=output_dir)

    output_file = output_dir / "fractal_noise.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump((signal_noise, params_scaled, params), f)
    print(f"Saved dataset: {output_file}")


if __name__ == "__main__":
    main()