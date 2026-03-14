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
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')

from utils.util_functions import Scale_params, signal_params, IVIM_model
from utils.perlin_and_pytorch import rand_perlin_2d_octaves

# uniform parameter distribution
def generator_uniform(Batch_size, patch, device, b_torch, bounds, snr):
    while True:
        # Generate IVIM params
        S0 = bounds[0,0] + (torch.rand((Batch_size, 1) + patch, device=device) * (bounds[1,0] - bounds[0,0]))
        Dt = bounds[0,1] + (torch.rand((Batch_size, 1) + patch, device=device) * (bounds[1,1] - bounds[0,1]))
        Fp = bounds[0,2] + (torch.rand((Batch_size, 1) + patch, device=device) * (bounds[1,2] - bounds[0,2]))
        Dp = bounds[0,3] + (torch.rand((Batch_size, 1) + patch, device=device) * (bounds[1,3] - bounds[0,3]))
        
        params = torch.cat((S0, Dt, Fp, Dp), axis=1)
        params_scaled = Scale_params(params, bounds)
        signal_noise = signal_params(params, snr, b_torch)
        signal_noise = signal_noise.to(device=device, dtype=torch.float32)
        yield abs(signal_noise), params_scaled, params

# Generate random uniform patches
def patch_generator_random_uniform(batch_size,patch, device, b_torch, bounds, snr):
    while True:
        # Generate IVIM params (S0, D, D*, f)
        # Generate Random Data for Two Different "Tissues"
        xA = torch.rand((2,batch_size, 1, 1, 1), device=device)*1 
        dA = torch.clamp(torch.rand((2,batch_size, 1, 1, 1), device=device) * 3e-3, min=0)
        d_ivimA = torch.rand((2,batch_size, 1, 1, 1), device=device) * (100e-3 - 3e-3) + 3e-3
        fA = torch.clamp(torch.rand((2,batch_size, 1, 1, 1), device=device)*0.5,min=0)
        
        # Patch Size and Mask Initialization
        sz = patch**2
        mid = sz//2
        batch_mask = torch.zeros(batch_size,1,patch,patch, device=device)
        # Generates random indices that exclude the central pixel initially, 
        # sorts them, and inserts the central pixel’s index back in.
        inds = torch.argsort(torch.rand(batch_size, sz-1, device=device))
        inds[inds>=mid] += 1
        inds = torch.cat((inds, torch.ones(inds.shape[0],1,dtype=torch.int64, device=device)*mid), axis=1)
        # This defines inds2, which has adjusted indices and is batch-specific. 
        # These are the locations within each patch that will be masked.
        inds2 = inds + (torch.arange(batch_size)*sz)[:,None].to(device=device)
        # Creating and Setting the Mask
        # Randomly selects one position within each patch to mask and 
        # adjusts mask to select the correct positions.
        n = torch.randint(sz,(batch_size,), device=device)
        n2 = n + torch.arange(batch_size, device=device)*sz
        mask = torch.zeros(inds2.shape, device=device)
        mask.view(-1)[n2] = 1
        mask = torch.cumsum(mask,1)==1.
        # Updates batch_mask based on these chosen positions. 
        # At this point, batch_mask marks the location for each 
        # pixel that will represent "tissue type 1" vs. "tissue type 2".
        batch_mask.view(-1)[inds2[mask]] = 1
        #% Making IVIM masked maps
        S0 = batch_mask*xA[0] + (1-batch_mask)*xA[1]
        Fp = batch_mask*fA[0] + (1-batch_mask)*fA[1]
        Dt = batch_mask*dA[0] + (1-batch_mask)*dA[1]
        Dp = batch_mask*d_ivimA[0] + (1-batch_mask)*d_ivimA[1]
        # Create plts ...
        plot_val_params = False
        if plot_val_params:
            for k in range(10):
                plt.figure(), plt.imshow(S0[k,0].cpu(), cmap='jet'),plt.colorbar(), plt.title('D')
                plt.figure(), plt.imshow(Dt[k,0].cpu(), cmap='jet'),plt.colorbar(), plt.title('D')
                plt.figure(), plt.imshow(Fp[k,0].cpu(), cmap='jet'),plt.colorbar(), plt.title('F')
                plt.figure(), plt.imshow(Dp[k,0].cpu(), cmap='jet'),plt.colorbar(), plt.title('D*')
        # Concatenate IVIM parameters into a single tensor along the channel dimension
        params = torch.cat((S0, Dt, Fp, Dp), axis=1)

        # Scale parameters to specified bounds and compute the signal using parameterized function
        params_scaled = Scale_params(params, bounds)
        signal = signal_params(params, snr, b_torch, device)
        signal_noise = signal.to(device=device, dtype=torch.float32)
        del Dp, Dt, Fp, S0, mask
        yield abs(signal_noise), params_scaled, params
        
def patch_generator_random_gaussian(batch_size,patch, device, b_torch, bounds, snr):
    while True:
        import random
        def constrained_gaussuan(bound_low,bound_up,n_values,bound_low_uniform,bound_up_uniform):
            # Generates 'n_values' normally distributed samples within 'bound_low' and 'bound_up'
            # Only keeps samples within the additional constraints 'bound_low_uniform' and 'bound_up_uniform'.
            
            # Calculate the midpoint of the range [bound_low, bound_up]
            diff_half = (bound_up - bound_low)/2
            mean = bound_low + diff_half # Set mean to the midpoint of [bound_low, bound_up]
            std = diff_half # Set standard deviation to half the range to keep samples near the mean
            
            values = []  # Initialize list to store valid samples
            while len(values) < n_values:  # Loop until we have the required number of samples
                sample = torch.normal(mean, std, size=(1,))  # Generate a sample from the normal distribution
                # Check if the sample falls within the specified uniform bounds
                if sample >= bound_low_uniform and sample < bound_up_uniform:
                    values.append(sample)  # Add sample to the list if within bounds
    
            return(values)
        
        # IVIM Parameters for Different Tissue Types
        S0_GM = constrained_gaussuan(0.2,0.5,batch_size,0,1)
        D_GM = constrained_gaussuan(0.0005,0.002,batch_size,0,3e-3)
        F_GM = constrained_gaussuan(0.05,0.25,batch_size,0,0.5)
        Dp_GM = constrained_gaussuan(0.003,0.03,batch_size,3e-3,100e-3)

        S0_WM = constrained_gaussuan(0.05,0.25,batch_size,0,1)
        D_WM = constrained_gaussuan(0.0003,0.0013,batch_size,0,3e-3)
        F_WM = constrained_gaussuan(0.01,0.1,batch_size,0,0.5)
        Dp_WM = constrained_gaussuan(0.003,0.015,batch_size,3e-3,100e-3)
        
        S0_CSF = constrained_gaussuan(0.5,1,batch_size,0,1)
        D_CSF = constrained_gaussuan(0.001,0.003,batch_size,0,3e-3)
        F_CSF = constrained_gaussuan(0.25,0.5,batch_size,0,0.5)
        Dp_CSF = constrained_gaussuan(0.02,0.05,batch_size,3e-3,100e-3)
        
        
        # Initialize lists to store parameter values for two tissue regions (A and B)
        xA, dA, d_ivimA, fA = [], [], [], []
        xB, dB, d_ivimB, fB = [], [], [], []

        # Randomly assign a tissue type (0, 1, or 2) for each batch element
        arr = torch.randint(0, 3, (batch_size,))

        # Helper function to append parameters for a given tissue type to lists A or B
        def append_params(tissue, i, x_list, d_list, d_ivim_list, f_list):
            if tissue == "GM":  # Grey Matter
                x_list.append(S0_GM[i])
                d_list.append(D_GM[i])
                d_ivim_list.append(Dp_GM[i])
                f_list.append(F_GM[i])
            elif tissue == "WM":  # White Matter
                x_list.append(S0_WM[i])
                d_list.append(D_WM[i])
                d_ivim_list.append(Dp_WM[i])
                f_list.append(F_WM[i])
            elif tissue == "CSF":  # Cerebrospinal Fluid
                x_list.append(S0_CSF[i])
                d_list.append(D_CSF[i])
                d_ivim_list.append(Dp_CSF[i])
                f_list.append(F_CSF[i])

        # Loop over each sample in the batch to define parameters based on the assigned tissue type
        for i in range(batch_size):
            arr_i = arr[i]  # Current tissue type (0 for GM, 1 for WM, 2 for CSF)
            
            # For Grey Matter (GM) as the main tissue type
            if arr_i == 0:
                append_params("GM", i, xA, dA, d_ivimA, fA)  # Main tissue type A is GM
                # Randomly assign White Matter (WM) or CSF as the secondary tissue type B
                arr2 = torch.randint(1, 3, (1,))
                if arr2 == 1:
                    append_params("WM", i, xB, dB, d_ivimB, fB)
                else:
                    append_params("CSF", i, xB, dB, d_ivimB, fB)

            # For White Matter (WM) as the main tissue type
            elif arr_i == 1:
                append_params("WM", i, xA, dA, d_ivimA, fA)  # Main tissue type A is WM
                # Randomly select Grey Matter (GM) or CSF as the secondary tissue type B
                arr2 = torch.tensor([0, 2])  # GM or CSF
                arr3 = torch.randint(0, 2, (1,))
                arr4 = arr2[arr3[0]]
                if arr4 == 0:
                    append_params("GM", i, xB, dB, d_ivimB, fB)
                else:
                    append_params("CSF", i, xB, dB, d_ivimB, fB)

            # For CSF (Cerebrospinal Fluid) as the main tissue type
            elif arr_i == 2:
                append_params("CSF", i, xA, dA, d_ivimA, fA)  # Main tissue type A is CSF
                # Randomly select Grey Matter (GM) or White Matter (WM) as the secondary tissue type B
                arr2 = torch.randint(0, 2, (1,))
                if arr2 == 0:
                    append_params("GM", i, xB, dB, d_ivimB, fB)
                else:
                    append_params("WM", i, xB, dB, d_ivimB, fB)
        
        # Define patch size and middle index for mask creation
        sz = patch**2               # Total number of elements in the patch
        mid = sz // 2               # Middle element index for the patch

        # Initialize a zero-filled batch mask with dimensions (batch_size, 1, patch, patch)
        batch_mask = torch.zeros(batch_size, 1, patch, patch, device=device)

        # Randomly sort indices for all but the middle voxel and adjust indices for each batch element
        inds = torch.argsort(torch.rand(batch_size, sz - 1, device=device))
        inds[inds >= mid] += 1      # Shift indices to avoid the middle voxel (reserve it for specific setting)
        inds = torch.cat((inds, torch.ones(inds.shape[0], 1, dtype=torch.int64, device=device) * mid), axis=1)

        # Adjust indices for batch elements by adding an offset for each batch index
        inds2 = inds + (torch.arange(batch_size) * sz)[:, None].to(device=device)

        # Select a random voxel to mask in each batch element
        n = torch.randint(sz, (batch_size,), device=device)
        n2 = n + torch.arange(batch_size, device=device) * sz

        # Initialize the mask with zeros and set chosen voxel indices to 1
        mask = torch.zeros(inds2.shape, device=device)
        mask.view(-1)[n2] = 1
        mask = torch.cumsum(mask, 1) == 1.
        batch_mask.view(-1)[inds2[mask]] = 1

        # Convert tissue parameters lists to tensors and add dimensions to match the patch size
        xA, xB = [torch.tensor(var, device=device)[:, None, None, None] for var in (xA, xB)]
        dA, dB = [torch.tensor(var, device=device)[:, None, None, None] for var in (dA, dB)]
        d_ivimA, d_ivimB = [torch.tensor(var, device=device)[:, None, None, None] for var in (d_ivimA, d_ivimB)]
        fA, fB = [torch.tensor(var, device=device)[:, None, None, None] for var in (fA, fB)]

        # Create masked IVIM parameter maps by combining tissue parameters A and B based on batch_mask
        S0 = batch_mask * xA + (1 - batch_mask) * xB
        Fp = batch_mask * fA + (1 - batch_mask) * fB
        Dt = batch_mask * dA + (1 - batch_mask) * dB
        Dp = batch_mask * d_ivimA + (1 - batch_mask) * d_ivimB

        # Create plots..
        plot_val_params = False
        if plot_val_params:
            for k in range(0,20):
                # plt.figure(), plt.imshow(S0[k,0].cpu(), cmap='jet'),plt.colorbar(), plt.title('S0')
                plt.figure(), plt.imshow(Dt[k,0].cpu(), cmap='jet', vmin=0, vmax=0.003 ),plt.colorbar(), plt.title('D')
                # plt.savefig(f'D_{k}')
                # plt.figure(), plt.imshow(Fp[k,0].cpu(), cmap='jet'),plt.colorbar(), plt.title('F')
                # plt.figure(), plt.imshow(Dp[k,0].cpu(), cmap='jet'),plt.colorbar(), plt.title('D*')
        # breakpoint()
        # Concatenate IVIM parameters into a single tensor along the channel dimension
        params = torch.cat((S0, Dt, Fp, Dp), axis=1)
        # Scale parameters to specified bounds and compute the signal using parameterized function
        params_scaled = Scale_params(params, bounds)
        signal = signal_params(params, snr, b_torch, device)
        signal_noise = signal.to(device=device, dtype=torch.float32)
        
        del Dp, Dt, Fp, S0, mask

        yield abs(signal_noise), params_scaled, params
        


def patch_generator_structured_uniform(batch_size,patch, device, b_torch, bounds, snr):
    while True:
        # Generate a fractal noise map for tissue-type segmentation and normalize it
        fractal_noise_map = rand_perlin_2d_octaves((256, 256), (8, 8), device, 4)
        fractal_noise_map = (fractal_noise_map - fractal_noise_map.min()) / fractal_noise_map.max()
        
        # Define binary masks for each tissue type based on fractal noise values
        # Values in noise map are thresholded to define masks for CSF, GM, and WM regions
        fractal_noise_map_CSF_mask = (fractal_noise_map < 0.2) | (fractal_noise_map > 0.8)
        fractal_noise_map_GM_mask = (fractal_noise_map < 0.5) & (fractal_noise_map > 0.2)
        fractal_noise_map_WM_mask = (fractal_noise_map < 0.8) & (fractal_noise_map > 0.5)

        # Assign integer labels to create a combined mask where each tissue type has a unique value
        combined_mask = (
            fractal_noise_map_WM_mask * 1 +
            fractal_noise_map_GM_mask * 2 +
            fractal_noise_map_CSF_mask * 3
        )
        
        # Reshape the combined mask into patches and adjust dimensions for batch processing
        # The reshaped mask allows each batch element to represent one tissue type with unique labels
        length_fm = patch * 12
        reshaped_mask = (
            combined_mask[:length_fm, :length_fm]
            .reshape(12, patch, 12, patch)
            .permute(0, 2, 1, 3)
            .reshape(144, patch, patch)[:batch_size, None]
        )

        # Define random IVIM parameters for three different tissue types
        xA = torch.rand((3, reshaped_mask.shape[0], 1, 1, 1), device=device) * 1
        dA = torch.clamp(torch.rand((3, reshaped_mask.shape[0], 1, 1, 1), device=device) * 3e-3, min=0)
        d_ivimA = torch.rand((3, reshaped_mask.shape[0], 1, 1, 1), device=device) * (100e-3 - 3e-3) + 3e-3
        fA = torch.clamp(torch.rand((3, reshaped_mask.shape[0], 1, 1, 1), device=device) * 0.5, min=0)

        # Create boolean masks for each tissue type (background, WM, GM, CSF)
        mask0, mask1, mask2 = (reshaped_mask == 0), (reshaped_mask == 1), (reshaped_mask == 2)

        # Generate IVIM parameter maps by combining values according to tissue-type masks
        S0 = mask0 * xA[0] + mask1 * xA[1] + mask2 * xA[2]
        Dt = mask0 * dA[0] + mask1 * dA[1] + mask2 * dA[2]
        Fp = mask0 * fA[0] + mask1 * fA[1] + mask2 * fA[2]
        Dp = mask0 * d_ivimA[0] + mask1 * d_ivimA[1] + mask2 * d_ivimA[2]

        # Plot the generated maps for visual inspection if plotting is enabled
        plot_figs = False
        if plot_figs:
            for i in range(reshaped_mask.shape[0]):
                # plt.figure(), plt.imshow(S0[i, 0].cpu(), cmap='jet'), plt.colorbar(), plt.title('S0')
                plt.figure(), plt.imshow(Dp[i, 0].cpu(), cmap='jet'), plt.colorbar(), plt.title('Dp')
                plt.savefig(f'Dp_{i}.png')
                # plt.figure(), plt.imshow(Dt[i, 0].cpu(), cmap='jet'), plt.colorbar(), plt.title('D')
                # plt.figure(), plt.imshow(Fp[i, 0].cpu(), cmap='jet'), plt.colorbar(), plt.title('F')
        
        # Concatenate the parameter maps along the channel dimension to create a multi-channel IVIM tensor
        params = torch.cat((S0, Dt, Fp, Dp), axis=1)

        # Scale the parameters and generate IVIM signal with added noise
        params_scaled = Scale_params(params, bounds)
        signal = signal_params(params, snr, b_torch, device)
        signal_noise = signal.to(dtype=torch.float32)

        # Yield the noisy signal, scaled parameters, and original parameters for further processing
        yield abs(signal_noise), params_scaled, params

def patch_generator_structured_gaussian(batch_size,patch, device, b_torch, bounds, snr):
    while True:
        # Initialize batch size for data generation
        batch_size = 128

        # Define a function to generate Gaussian-distributed values within specified bounds
        def constrained_gaussuan(bound_low, bound_up, n_values, bound_low_uniform, bound_up_uniform):
            """
            Generate random values from a normal distribution, constrained within specified bounds.
            
            Parameters:
                bound_low (float): Lower bound for Gaussian distribution.
                bound_up (float): Upper bound for Gaussian distribution.
                n_values (int): Number of values to generate.
                bound_low_uniform (float): Lower bound for uniform constraint.
                bound_up_uniform (float): Upper bound for uniform constraint.
            
            Returns:
                values (list): A list of values that fall within both the Gaussian and uniform constraints.
            """
            # Calculate half the difference between the upper and lower bounds for the distribution
            diff_half = (bound_up - bound_low) / 2
            mean = bound_low + diff_half  # Mean of the Gaussian distribution
            std = diff_half  # Standard deviation of the Gaussian distribution
            values = []
            
            # Sample values until the specified number of values (n_values) is obtained
            while len(values) < n_values:
                sample = torch.normal(mean, std, size=(1,))
                # Ensure the sampled value is within the uniform constraints
                if sample >= bound_low_uniform and sample < bound_up_uniform:
                    values.append(sample)
            return values

        # Generate tissue-specific Gaussian parameters for different tissues (GM, WM, CSF)
        S0_GM = constrained_gaussuan(0.2, 0.5, 128, 0, 1)
        D_GM = constrained_gaussuan(0.0005, 0.002, 128, 0, 3e-3)
        F_GM = constrained_gaussuan(0.05, 0.25, 128, 0, 0.5)
        Dp_GM = constrained_gaussuan(0.003, 0.03, 128, 3e-3, 100e-3)

        S0_WM = constrained_gaussuan(0.05, 0.25, 128, 0, 1)
        D_WM = constrained_gaussuan(0.0003, 0.0013, 128, 0, 3e-3)
        F_WM = constrained_gaussuan(0.01, 0.1, 128, 0, 0.5)
        Dp_WM = constrained_gaussuan(0.003, 0.015, 128, 3e-3, 100e-3)
        
        S0_CSF = constrained_gaussuan(0.5, 1, 128, 0, 1)
        D_CSF = constrained_gaussuan(0.001, 0.003, 128, 0, 3e-3)
        F_CSF = constrained_gaussuan(0.25, 0.5, 128, 0, 0.5)
        Dp_CSF = constrained_gaussuan(0.02, 0.05, 128, 3e-3, 100e-3)

        # Generate a fractal noise map to create tissue segmentation masks
        fractal_noise_map1_mask = rand_perlin_2d_octaves((256, 256), (8, 8), device, 4)
        
        # Normalize the fractal noise map between 0 and 1
        fractal_noise_map1_mask = fractal_noise_map1_mask - fractal_noise_map1_mask.min()
        fractal_noise_map1_mask = fractal_noise_map1_mask / fractal_noise_map1_mask.max()

        # Create masks for different tissue types (CSF, GM, WM) based on the fractal noise map
        fractal_noise_map_CSF_mask = (fractal_noise_map1_mask < 0.2) | (fractal_noise_map1_mask > 0.8)
        fractal_noise_map_GM_mask = (fractal_noise_map1_mask < 0.5) & (fractal_noise_map1_mask > 0.2)
        fractal_noise_map_WM_mask = (fractal_noise_map1_mask < 0.8) & (fractal_noise_map1_mask > 0.5)

        # Calculate the size of the fractal noise map for the tissue mask and reshape it for use in the batch
        length_fm = patch * 12
        new_mask = fractal_noise_map_WM_mask * 1 + fractal_noise_map_GM_mask * 2 + fractal_noise_map_CSF_mask * 3
        reshaped_mask = new_mask[:length_fm, :length_fm].reshape(12, patch, 12, patch).permute(0, 2, 1, 3).reshape(144, patch, patch)[:batch_size, None]

        # Create binary masks for each tissue type (GM, WM, CSF)
        maskGM = (reshaped_mask == 1)
        maskWM = (reshaped_mask == 2)
        maskCSF = (reshaped_mask == 3)

        # Convert the generated Gaussian parameters into tensors for each tissue type
        S0_GM_array = torch.tensor(S0_GM, device=device)[:, None, None, None]
        S0_WM_array = torch.tensor(S0_WM, device=device)[:, None, None, None]
        S0_CSF_array = torch.tensor(S0_CSF, device=device)[:, None, None, None]

        D_GM_array = torch.tensor(D_GM, device=device)[:, None, None, None]
        D_WM_array = torch.tensor(D_WM, device=device)[:, None, None, None]
        D_CSF_array = torch.tensor(D_CSF, device=device)[:, None, None, None]

        F_GM_array = torch.tensor(F_GM, device=device)[:, None, None, None]
        F_WM_array = torch.tensor(F_WM, device=device)[:, None, None, None]
        F_CSF_array = torch.tensor(F_CSF, device=device)[:, None, None, None]

        Dp_GM_array = torch.tensor(Dp_GM, device=device)[:, None, None, None]
        Dp_WM_array = torch.tensor(Dp_WM, device=device)[:, None, None, None]
        Dp_CSF_array = torch.tensor(Dp_CSF, device=device)[:, None, None, None]

        # Apply tissue-specific masks to assign the appropriate values to each tissue's parameter maps
        S0 = maskGM * S0_GM_array + maskWM * S0_WM_array + maskCSF * S0_CSF_array
        Dt = maskGM * D_GM_array + maskWM * D_WM_array + maskCSF * D_CSF_array
        Fp = maskGM * F_GM_array + maskWM * F_WM_array + maskCSF * F_CSF_array
        Dp = maskGM * Dp_GM_array + maskWM * Dp_WM_array + maskCSF * Dp_CSF_array

        # Optional: Visualize the generated Dp maps for debugging or inspection
        plot_figs = False
        if plot_figs:
            for i in range(0, 20):
                plt.figure(), plt.imshow(Dp[i, 0].cpu(), cmap='jet'), plt.colorbar(), plt.title('Dp')
                plt.savefig(f'Dp_{i}.png')  # Save the plots as images

        # Concatenate the tissue parameter maps into a single tensor
        params = torch.cat((S0, Dt, Fp, Dp), axis=1)

        # Scale the parameters to the specified bounds (normalization or scaling process)
        params_scaled = Scale_params(params, bounds)

        # Generate the signal using the parameters and the given signal-to-noise ratio (SNR)
        signal = signal_params(params, snr, b_torch, device)
        signal_noise = signal.to(dtype=torch.float32)

        # Yield the signal with noise, the scaled parameters, and the original parameters for further processing
        yield abs(signal_noise), params_scaled, params


def patch_generator_perlin_test(batch_size, patch, device, b_torch, bounds, snr):
    while True:
        # Tissue map generation using fractal noise maps for different tissue types (GM, WM, CSF)
        
        # Generate fractal noise maps for various properties (S0, D, F, Dp) across different tissue types
        def generate_fractal_noise_map(device):
            """Generate and normalize a 2D fractal noise map."""
            noise_map = rand_perlin_2d_octaves((256, 256), (8, 8), device, 4)
            noise_map = (noise_map - noise_map.min()) / (noise_map.max() - noise_map.min())
            return noise_map

        # Create noise maps for GM (Gray Matter), WM (White Matter), CSF (Cerebrospinal Fluid)
        fractal_noise_map1_mask = generate_fractal_noise_map(device)  # For GM, WM, CSF mask generation
        fractal_noise_S0map1 = generate_fractal_noise_map(device)     # S0 for GM
        fractal_noise_S0map2 = generate_fractal_noise_map(device)     # S0 for WM
        fractal_noise_S0map3 = generate_fractal_noise_map(device)     # S0 for CSF

        fractal_noise_Dmap1 = generate_fractal_noise_map(device)      # D for GM
        fractal_noise_Dmap2 = generate_fractal_noise_map(device)      # D for WM
        fractal_noise_Dmap3 = generate_fractal_noise_map(device)      # D for CSF

        fractal_noise_Fmap1 = generate_fractal_noise_map(device)      # F for GM
        fractal_noise_Fmap2 = generate_fractal_noise_map(device)      # F for WM
        fractal_noise_Fmap3 = generate_fractal_noise_map(device)      # F for CSF

        fractal_noise_Dpmap1 = generate_fractal_noise_map(device)     # Dp for GM
        fractal_noise_Dpmap2 = generate_fractal_noise_map(device)     # Dp for WM
        fractal_noise_Dpmap3 = generate_fractal_noise_map(device)     # Dp for CSF

        # Create masks for different tissue types (CSF, GM, WM) based on fractal noise
        fractal_noise_map_CSF_mask = (fractal_noise_map1_mask < 0.2) + (fractal_noise_map1_mask > 0.8)
        fractal_noise_map_GM_mask = (fractal_noise_map1_mask < 0.5) & (fractal_noise_map1_mask > 0.2)
        fractal_noise_map_WM_mask = (fractal_noise_map1_mask < 0.8) & (fractal_noise_map1_mask > 0.5)

        # Scaling of fractal noise maps to match specific value ranges for each tissue property
        fractal_noise_S0_GM = 0.2 + fractal_noise_S0map1 * (0.5 - 0.2)
        fractal_noise_D_GM = 0.0005 + fractal_noise_Dmap1 * (0.002 - 0.0005)
        fractal_noise_F_GM = 0.05 + fractal_noise_Fmap1 * (0.25 - 0.05)
        fractal_noise_Dp_GM = 0.003 + fractal_noise_Dpmap1 * (0.03 - 0.003)

        fractal_noise_S0_WM = 0.05 + fractal_noise_S0map2 * (0.25 - 0.05)
        fractal_noise_D_WM = 0.0003 + fractal_noise_Dmap2 * (0.0013 - 0.0003)
        fractal_noise_F_WM = 0.01 + fractal_noise_Fmap2 * (0.1 - 0.01)
        fractal_noise_Dp_WM = 0.003 + fractal_noise_Dpmap2 * (0.015 - 0.003)

        fractal_noise_S0_CSF = 0.5 + fractal_noise_S0map3 * (1 - 0.5)
        fractal_noise_D_CSF = 0.001 + fractal_noise_Dmap3 * (0.003 - 0.001)
        fractal_noise_F_CSF = 0.25 + fractal_noise_Fmap3 * (0.5 - 0.25)
        fractal_noise_Dp_CSF = 0.02 + fractal_noise_Dpmap3 * (0.05 - 0.02)

        # Combine fractal noise maps using the generated masks to obtain final S0, D, F, Dp maps for each tissue type
        fractal_noise_S0_map = fractal_noise_S0_GM * fractal_noise_map_GM_mask + fractal_noise_S0_WM * fractal_noise_map_WM_mask + fractal_noise_S0_CSF * fractal_noise_map_CSF_mask
        fractal_noise_D_map = fractal_noise_D_GM * fractal_noise_map_GM_mask + fractal_noise_D_WM * fractal_noise_map_WM_mask + fractal_noise_D_CSF * fractal_noise_map_CSF_mask
        fractal_noise_F_map = fractal_noise_F_GM * fractal_noise_map_GM_mask + fractal_noise_F_WM * fractal_noise_map_WM_mask + fractal_noise_F_CSF * fractal_noise_map_CSF_mask
        fractal_noise_Dp_map = fractal_noise_Dp_GM * fractal_noise_map_GM_mask + fractal_noise_Dp_WM * fractal_noise_map_WM_mask + fractal_noise_Dp_CSF * fractal_noise_map_CSF_mask

        # Set the patch size for the final image slices (12 patches per tissue type)
        length_fm = patch * 12  # Adjust patch size for final reshaping

        # Reshape the generated maps into desired tensor shapes for use in model (e.g., 144 patches per batch)
        def reshape_map_to_tensor(fractal_noise_map):
            return fractal_noise_map[:length_fm, :length_fm].reshape(12, patch, 12, patch).permute(0, 2, 1, 3).reshape(144, patch, patch)[:batch_size][:, None]

        # Apply reshaping to each of the noise maps
        S0 = reshape_map_to_tensor(fractal_noise_S0_map)
        Dt = reshape_map_to_tensor(fractal_noise_D_map)
        Fp = reshape_map_to_tensor(fractal_noise_F_map)
        Dp = reshape_map_to_tensor(fractal_noise_Dp_map)

        # Optionally plot the generated Dp maps for visualization
        plot_figs = False
        if plot_figs:
            for i in range(0, 10):
                plt.figure()
                plt.imshow(Dp[i, 0].cpu(), cmap='jet')
                plt.colorbar()
                plt.title('Dp')
                plt.savefig(f'Dp_new5_{i}.png')


        # Combine the individual tissue property maps into a single tensor
        params = torch.cat((S0, Dt, Fp, Dp), axis=1)

        # Scale the parameters to the required bounds
        params_scaled = Scale_params(params, bounds)

        # Generate the corresponding signal from the tissue parameters, based on the signal model
        signal = signal_params(params, snr, b_torch, device)

        # Add noise to the generated signal
        signal_noise = signal.to(dtype=torch.float32)

        # Return the signal and the corresponding parameters for further processing
        yield abs(signal_noise), params_scaled, params

        
def generator_perlin_noise(batch_size, patch, device, b_torch, bounds, snr):
    """
    Generates a batch of Perlin noise-based tissue maps with parameters and returns
    the generated signal, scaled parameters, and original parameters.
    """

    # Helper function to generate normalized Perlin noise map
    def generate_normalized_perlin_noise(shape, device):
        """
        Generates a Perlin noise map, normalizes it to the [0, 1] range.
        """
        fractal_noise = rand_perlin_2d_octaves(shape, (8, 8), device, 4)
        fractal_noise = fractal_noise - fractal_noise.min()
        return fractal_noise / fractal_noise.max()

    # Helper function to generate a tissue-specific map
    def generate_tissue_map(noise_map, min_val, max_val, mask):
        """
        Generates a tissue map by scaling the Perlin noise map to a specified range 
        and applying a mask for a specific tissue type.
        """
        return min_val + (noise_map * (max_val - min_val)) * mask

    # Helper function to combine tissue maps based on masks
    def combine_tissue_maps(gm_map, wm_map, csf_map, gm_mask, wm_mask, csf_mask):
        """
        Combines the GM, WM, and CSF tissue maps based on their respective masks.
        """
        return gm_map * gm_mask + wm_map * wm_mask + csf_map * csf_mask

    # Initialize tensors for S0, D, F, and Dp maps (these will store generated maps for each batch)
    S0 = torch.zeros((batch_size, 1, 128, 128))
    Dt = torch.zeros((batch_size, 1, 128, 128))
    Fp = torch.zeros((batch_size, 1, 128, 128))
    Dp = torch.zeros((batch_size, 1, 128, 128))

    # Loop over each batch to generate the corresponding tissue map
    for i in range(batch_size):

        # Generate various Perlin noise maps with different fractal characteristics
        fractal_noise_map1_mask = generate_normalized_perlin_noise((256, 256), device)
        fractal_noise_S0map1 = generate_normalized_perlin_noise((256, 256), device)
        fractal_noise_S0map2 = generate_normalized_perlin_noise((256, 256), device)
        fractal_noise_S0map3 = generate_normalized_perlin_noise((256, 256), device)
        
        fractal_noise_Dmap1 = generate_normalized_perlin_noise((256, 256), device)
        fractal_noise_Dmap2 = generate_normalized_perlin_noise((256, 256), device)
        fractal_noise_Dmap3 = generate_normalized_perlin_noise((256, 256), device)
        
        fractal_noise_Fmap1 = generate_normalized_perlin_noise((256, 256), device)
        fractal_noise_Fmap2 = generate_normalized_perlin_noise((256, 256), device)
        fractal_noise_Fmap3 = generate_normalized_perlin_noise((256, 256), device)
        
        fractal_noise_Dpmap1 = generate_normalized_perlin_noise((256, 256), device)
        fractal_noise_Dpmap2 = generate_normalized_perlin_noise((256, 256), device)
        fractal_noise_Dpmap3 = generate_normalized_perlin_noise((256, 256), device)

        # Masks for different tissue types (CSF, GM, WM) based on the noise map
        fractal_noise_map_CSF_mask = (fractal_noise_map1_mask < 0.2) + (fractal_noise_map1_mask > 0.8)
        fractal_noise_map_GM_mask = (fractal_noise_map1_mask < 0.5) & (fractal_noise_map1_mask > 0.2)
        fractal_noise_map_WM_mask = (fractal_noise_map1_mask < 0.8) & (fractal_noise_map1_mask > 0.5)

        # Generate tissue-specific maps by scaling the noise and applying the appropriate tissue masks
        fractal_noise_S0_GM = generate_tissue_map(fractal_noise_S0map1, 0.2, 0.5, fractal_noise_map_GM_mask)
        fractal_noise_D_GM = generate_tissue_map(fractal_noise_Dmap1, 0.0005, 0.002, fractal_noise_map_GM_mask)
        fractal_noise_F_GM = generate_tissue_map(fractal_noise_Fmap1, 0.05, 0.25, fractal_noise_map_GM_mask)
        fractal_noise_Dp_GM = generate_tissue_map(fractal_noise_Dpmap1, 0.003, 0.03, fractal_noise_map_GM_mask)

        fractal_noise_S0_WM = generate_tissue_map(fractal_noise_S0map2, 0.05, 0.25, fractal_noise_map_WM_mask)
        fractal_noise_D_WM = generate_tissue_map(fractal_noise_Dmap2, 0.0003, 0.0013, fractal_noise_map_WM_mask)
        fractal_noise_F_WM = generate_tissue_map(fractal_noise_Fmap2, 0.01, 0.1, fractal_noise_map_WM_mask)
        fractal_noise_Dp_WM = generate_tissue_map(fractal_noise_Dpmap2, 0.003, 0.015, fractal_noise_map_WM_mask)

        fractal_noise_S0_CSF = generate_tissue_map(fractal_noise_S0map3, 0.5, 1.0, fractal_noise_map_CSF_mask)
        fractal_noise_D_CSF = generate_tissue_map(fractal_noise_Dmap3, 0.001, 0.003, fractal_noise_map_CSF_mask)
        fractal_noise_F_CSF = generate_tissue_map(fractal_noise_Fmap3, 0.25, 0.5, fractal_noise_map_CSF_mask)
        fractal_noise_Dp_CSF = generate_tissue_map(fractal_noise_Dpmap3, 0.02, 0.05, fractal_noise_map_CSF_mask)

        # Combine the individual tissue maps based on the corresponding tissue masks
        fractal_noise_S0_map = combine_tissue_maps(fractal_noise_S0_GM, fractal_noise_S0_WM, fractal_noise_S0_CSF, 
                                                   fractal_noise_map_GM_mask, fractal_noise_map_WM_mask, fractal_noise_map_CSF_mask)
        fractal_noise_D_map = combine_tissue_maps(fractal_noise_D_GM, fractal_noise_D_WM, fractal_noise_D_CSF, 
                                                   fractal_noise_map_GM_mask, fractal_noise_map_WM_mask, fractal_noise_map_CSF_mask)
        fractal_noise_F_map = combine_tissue_maps(fractal_noise_F_GM, fractal_noise_F_WM, fractal_noise_F_CSF, 
                                                   fractal_noise_map_GM_mask, fractal_noise_map_WM_mask, fractal_noise_map_CSF_mask)
        fractal_noise_Dp_map = combine_tissue_maps(fractal_noise_Dp_GM, fractal_noise_Dp_WM, fractal_noise_Dp_CSF, 
                                                   fractal_noise_map_GM_mask, fractal_noise_map_WM_mask, fractal_noise_map_CSF_mask)

        # Assign the generated maps to the batch tensors (downsampling to 128x128)
        S0[i, 0] = fractal_noise_S0_map[:128, :128]
        Dt[i, 0] = fractal_noise_D_map[:128, :128]
        Fp[i, 0] = fractal_noise_F_map[:128, :128]
        Dp[i, 0] = fractal_noise_Dp_map[:128, :128]

    # Optionally plot and save generated fractal maps (for debugging/visualization purposes)
    plot_figs = True
    if plot_figs:
        for i in range(0, 10):
            plt.figure(), plt.imshow(Dp[i, 0].cpu(), cmap='jet'), plt.colorbar(), plt.title('Dp')
            plt.savefig(f'fractalnoise_{i}.png')

    # Concatenate generated maps into a single tensor for return
    params = torch.cat((S0, Dt, Fp, Dp), axis=1)

    # Apply scaling to the parameters and generate signals with added noise
    params_scaled = Scale_params(params, bounds)
    signal = signal_params(params, snr, b_torch, device)
    signal_noise = signal.to(dtype=torch.float32)

    # Yield the generated signal with noise, scaled parameters, and original parameters
    yield abs(signal_noise), params_scaled, params


def generator_perlin_noise_patch(batch_size, image_shape, device, b_torch, bounds, snr, res, octaves):
    """
    Generates a batch of Perlin noise-based tissue maps with parameters and returns
    the generated signal, scaled parameters, and original parameters.
    """

    # Helper function to generate normalized Perlin noise map
    def generate_normalized_perlin_noise(shape, device, res = 4, octaves = 4):
        """
        Generates a Perlin noise map, normalizes it to the [0, 1] range.
        """
        # Change (shape, res, device, octaves) if necessary. Note resolution 
        # and octaves increase and decrease the number of fractal-noise instances 
        # in the map
        fractal_noise = rand_perlin_2d_octaves(shape, (res, res), device, octaves)
        fractal_noise = fractal_noise - fractal_noise.min()
        return fractal_noise / fractal_noise.max()

    # Helper function to generate a tissue-specific map
    def generate_tissue_map(noise_map, min_val, max_val, mask):
        """
        Generates a tissue map by scaling the Perlin noise map to a specified range 
        and applying a mask for a specific tissue type.
        """
        return min_val + (noise_map * (max_val - min_val)) * mask

    # Helper function to combine tissue maps based on masks
    def combine_tissue_maps(gm_map, wm_map, csf_map, gm_mask, wm_mask, csf_mask):
        """
        Combines the GM, WM, and CSF tissue maps based on their respective masks.
        """
        return gm_map * gm_mask + wm_map * wm_mask + csf_map * csf_mask

    # Initialize tensors for S0, D, F, and Dp maps (these will store generated maps for each batch)
    S0 = torch.zeros((batch_size, 1, image_shape[0], image_shape[1]))
    Dt = torch.zeros((batch_size, 1, image_shape[0], image_shape[1]))
    Fp = torch.zeros((batch_size, 1, image_shape[0], image_shape[1]))
    Dp = torch.zeros((batch_size, 1, image_shape[0], image_shape[1]))

    # Loop over each batch to generate the corresponding tissue map
    for i in range(batch_size):

        # Generate various Perlin noise maps with different fractal characteristics
        fractal_noise_map1_mask = generate_normalized_perlin_noise((image_shape[0], image_shape[1]), device, res, octaves)
        fractal_noise_S0map1 = generate_normalized_perlin_noise((image_shape[0], image_shape[1]), device, res, octaves)
        fractal_noise_S0map2 = generate_normalized_perlin_noise((image_shape[0], image_shape[1]), device, res, octaves)
        fractal_noise_S0map3 = generate_normalized_perlin_noise((image_shape[0], image_shape[1]), device, res, octaves)
        
        fractal_noise_Dmap1 = generate_normalized_perlin_noise((image_shape[0], image_shape[1]), device, res, octaves)
        fractal_noise_Dmap2 = generate_normalized_perlin_noise((image_shape[0], image_shape[1]), device, res, octaves)
        fractal_noise_Dmap3 = generate_normalized_perlin_noise((image_shape[0], image_shape[1]), device, res, octaves)
        
        fractal_noise_Fmap1 = generate_normalized_perlin_noise((image_shape[0], image_shape[1]), device, res, octaves)
        fractal_noise_Fmap2 = generate_normalized_perlin_noise((image_shape[0], image_shape[1]), device, res, octaves)
        fractal_noise_Fmap3 = generate_normalized_perlin_noise((image_shape[0], image_shape[1]), device, res, octaves)
        
        fractal_noise_Dpmap1 = generate_normalized_perlin_noise((image_shape[0], image_shape[1]), device, res, octaves)
        fractal_noise_Dpmap2 = generate_normalized_perlin_noise((image_shape[0], image_shape[1]), device, res, octaves)
        fractal_noise_Dpmap3 = generate_normalized_perlin_noise((image_shape[0], image_shape[1]), device, res, octaves)

        # Masks for different tissue types (CSF, GM, WM) based on the noise map
        fractal_noise_map_CSF_mask = (fractal_noise_map1_mask < 0.2) + (fractal_noise_map1_mask > 0.8)
        fractal_noise_map_GM_mask = (fractal_noise_map1_mask < 0.5) & (fractal_noise_map1_mask > 0.2)
        fractal_noise_map_WM_mask = (fractal_noise_map1_mask < 0.8) & (fractal_noise_map1_mask > 0.5)

        # Generate tissue-specific maps by scaling the noise and applying the appropriate tissue masks
        fractal_noise_S0_GM = generate_tissue_map(fractal_noise_S0map1, 0.2, 0.5, fractal_noise_map_GM_mask)
        fractal_noise_D_GM = generate_tissue_map(fractal_noise_Dmap1, 0.0005, 0.002, fractal_noise_map_GM_mask)
        fractal_noise_F_GM = generate_tissue_map(fractal_noise_Fmap1, 0.05, 0.25, fractal_noise_map_GM_mask)
        fractal_noise_Dp_GM = generate_tissue_map(fractal_noise_Dpmap1, 0.003, 0.03, fractal_noise_map_GM_mask)

        fractal_noise_S0_WM = generate_tissue_map(fractal_noise_S0map2, 0.05, 0.25, fractal_noise_map_WM_mask)
        fractal_noise_D_WM = generate_tissue_map(fractal_noise_Dmap2, 0.0003, 0.0013, fractal_noise_map_WM_mask)
        fractal_noise_F_WM = generate_tissue_map(fractal_noise_Fmap2, 0.01, 0.1, fractal_noise_map_WM_mask)
        fractal_noise_Dp_WM = generate_tissue_map(fractal_noise_Dpmap2, 0.003, 0.015, fractal_noise_map_WM_mask)

        fractal_noise_S0_CSF = generate_tissue_map(fractal_noise_S0map3, 0.5, 1.0, fractal_noise_map_CSF_mask)
        fractal_noise_D_CSF = generate_tissue_map(fractal_noise_Dmap3, 0.001, 0.003, fractal_noise_map_CSF_mask)
        fractal_noise_F_CSF = generate_tissue_map(fractal_noise_Fmap3, 0.25, 0.5, fractal_noise_map_CSF_mask)
        fractal_noise_Dp_CSF = generate_tissue_map(fractal_noise_Dpmap3, 0.02, 0.05, fractal_noise_map_CSF_mask)

        # Combine the individual tissue maps based on the corresponding tissue masks
        fractal_noise_S0_map = combine_tissue_maps(fractal_noise_S0_GM, fractal_noise_S0_WM, fractal_noise_S0_CSF, 
                                                   fractal_noise_map_GM_mask, fractal_noise_map_WM_mask, fractal_noise_map_CSF_mask)
        fractal_noise_D_map = combine_tissue_maps(fractal_noise_D_GM, fractal_noise_D_WM, fractal_noise_D_CSF, 
                                                   fractal_noise_map_GM_mask, fractal_noise_map_WM_mask, fractal_noise_map_CSF_mask)
        fractal_noise_F_map = combine_tissue_maps(fractal_noise_F_GM, fractal_noise_F_WM, fractal_noise_F_CSF, 
                                                   fractal_noise_map_GM_mask, fractal_noise_map_WM_mask, fractal_noise_map_CSF_mask)
        fractal_noise_Dp_map = combine_tissue_maps(fractal_noise_Dp_GM, fractal_noise_Dp_WM, fractal_noise_Dp_CSF, 
                                                   fractal_noise_map_GM_mask, fractal_noise_map_WM_mask, fractal_noise_map_CSF_mask)

        # Assign the generated maps to the batch tensors (downsampling to 128x128)
        S0[i, 0] = fractal_noise_S0_map
        Dt[i, 0] = fractal_noise_D_map
        Fp[i, 0] = fractal_noise_F_map
        Dp[i, 0] = fractal_noise_Dp_map

    # Optionally plot and save generated fractal maps (for debugging/visualization purposes)
    plot_figs = False
    if plot_figs:
        for i in range(0, 10):
            plt.figure(), plt.imshow(Dp[i, 0].cpu(), cmap='jet'), plt.colorbar(), plt.title('Dp')
            plt.savefig(f'fractalnoise_{i}.png')

    # Concatenate generated maps into a single tensor for return
    params = torch.cat((S0, Dt, Fp, Dp), axis=1)

    # Apply scaling to the parameters and generate signals with added noise
    params_scaled = Scale_params(params, bounds)
    signal = signal_params(params, snr, b_torch, device)
    signal_noise = signal.to(dtype=torch.float32)

    # Yield the generated signal with noise, scaled parameters, and original parameters
    yield abs(signal_noise), params_scaled, params

        
