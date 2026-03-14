# Incorporating spatial information in deep learning parameter estimation
This repository contains the code for our transformer-based deep learning model fitting approaches described in the following publications:
- Incorporating spatial information in deep learning parameter estimation with application to the intravoxel incoherent motion model in diffusion-weighted MRI [MEDIA paper](https://www.sciencedirect.com/science/article/pii/S1361841524003396)
- A Comparative Study of IVIM-MRI Fitting Techniques in Glioma Grading: Conventional, Bayesian, and Voxel-wise and Spatially-Aware Deep Learning Approaches [JMRI paper](https://).

The work focuses on improving parameter estimation in diffusion MRI by incorporating neighborhood information using transformer architectures. The models are trained on synthetic data patches, addressing limitations of traditional voxel-wise fitting methods.

While the implementation in this repository focuses on IVIM parameter estimation, the framework can be adapted to other model-fitting problems in diffusion MRI or related imaging modalities.

## 🛠️ Setup 

Make sure pip packaging tools are up-to-date: ``` pip install --upgrade pip setuptools wheel ```

Install packages: ``` pip install -r requirements.txt ```

### 🚀 Run

Running ``` transformer_parameter_estimation.sh ``` does the following:
- Generate fractal-noise-based parameter maps.
- Trains a transformer (SA-17 or NATTEN-17) on patches of model parameters.
- Performs inference using the trained transformer on the generated parameter maps.

For altering training, network and simulations parameters, please alter  ``` config.py ```


We also provide an ``` invivo_helpers.py ``` script, which can be used to normalize in vivo diffusion MRI data to match the value range used during training (i.e., values between 0 and 1). This required an in vivo diffusion MRI image with dimensions (x, y, z, b) and a homogeneous tissue mask (x, y, z) (e.g., a white matter mask). This normalization ensures compatibility between real data and the synthetic training distribution.

## 📜 Citation

- Incorporating spatial information in deep learning parameter estimation with application to the intravoxel incoherent motion model in diffusion-weighted MRI [MEDIA paper](https://www.sciencedirect.com/science/article/pii/S1361841524003396)
- A Comparative Study of IVIM-MRI Fitting Techniques in Glioma Grading: Conventional, Bayesian, and Voxel-wise and Spatially-Aware Deep Learning Approaches [JMRI paper](https://).

## 📬 Contact
For questions or issues, please contact:  
misha.kaandorp@kispi.uzh.ch

