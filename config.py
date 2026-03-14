# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 20:59:39 2024

@author: Misha Kaandorp
Publication: 'Incorporating spatial information in deep learning parameter estimation 
with application to the intravoxel incoherent motion model in diffusion-weighted MRI'

Publication: 'A Comparative Study of IVIM-MRI Fitting Techniques in Glioma Grading: 
Conventional, Bayesian, and Voxel-Wise and Spatially-Aware Deep Learning Approaches'
"""

import shutil
from pathlib import Path
from datetime import datetime

# Configuration dictionary

CONFIG = {

    "model_name": "SA-17",

    "paths": {
        "output_dir": "runs/run_1",
        "resume_weight": ""
    },

    "device": "gpu",

    "training": {
        "training_generator": "patch_generator_random_uniform",
        "training_strategy": "supervised",
        "learning_rate": 1e-4,
        "n_epochs": 3000,  # can also be lower.. 300 .. Please ensure convergence of the network before evaluation
        "batches_per_epoch": 500,
        "batch_size": 128,
        "patch_size": 17  # patch_size of the generator
    },

    "dataset": {
        # Choice of dataset generator: options could be "generator_perlin_noise_patch", etc.
        "generator": "generator_perlin_noise_patch",
        "device": "cpu",  # optional, can override GPU if desired
        "GT_and_error_maps_flag": True,
        "fractal_noise": {
            "batch_size": 40,
            "image_shape": [128, 128],
            "res": 4,
            "octaves": 4
        }
    },

    "ivim": {
        "snr": 100,  # This defines that S0 = 1 where 1 is equal to snr=100
        "bvalues": [
            0, 10, 20, 40, 80, 110, 140, 170,
            200, 300, 400, 500, 600, 700, 800, 900
        ],
        "bounds": [
            [0, 0, 0, 3e-3],
            [1, 3e-3, 0.5, 100e-3]  # S0 D f D*
        ]
    },

    "models": {
        "SA-17": {
            "hidden_channels": 128,
            "num_layers": 3,
            "kernel_NATTEN": 3,
            "norm_first": True,
            "attention": "SELF",
            "out_channels": 4,
        },
        "NATTEN-17": {
            "hidden_channels": 128,
            "num_layers": 8,
            "kernel_NATTEN": 3,
            "norm_first": True,
            "attention": "NATTEN",
            "out_channels": 4
        }
    },
}

# Copy this config file to the output directory

try:
    output_dir = Path(CONFIG["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    current_file = Path(__file__).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination_file = output_dir / f"{current_file.stem}.txt"

    shutil.copy(current_file, destination_file)
    print(f"Config file copied to {destination_file}")

except Exception as e:
    print(f"Failed to copy config file: {e}")