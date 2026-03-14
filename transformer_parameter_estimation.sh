# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 20:59:39 2024

@author: Misha Kaandorp
Publication: 'Incorporating spatial information in deep learning parameter estimation 
with application to the intravoxel incoherent motion model in diffusion-weighted MRI'

Publication: 'A Comparative Study of IVIM-MRI Fitting Techniques in Glioma Grading: 
Conventional, Bayesian, and Voxel-Wise and Spatially-Aware Deep Learning Approaches'
"""

set -e

########################################
# 1. Synthesize fractal-noise parameter maps
########################################
synthesize_parameter_maps () {
    echo "Synthesizing parameter maps..."
    python generate_fractal_based_parameter_maps.py
}

########################################
# 2. Train model
########################################
train_model () {
    echo "Training model..."
    python train.py
}

########################################
# 3. Run inference / evaluation
########################################
run_inference () {
    echo "Running inference..."
    python inference.py
}

########################################
# Main pipeline
########################################
main () {
    # Uncomment steps as needed
    synthesize_parameter_maps
    train_model
    run_inference
}

main "$@"