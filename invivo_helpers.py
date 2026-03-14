import nibabel as nib
import numpy as np
from config import CONFIG


def scale_invivo_by_snr(
    ref_mri_path: str,
    ref_roi_path: str,
    B0_slice_index: int,
    save_path: str
) -> None:
    """
    Scale an in-vivo MRI image based on a reference ROI and desired SNR.

    Parameters
    ----------
    ref_mri_path : str
        Path to the reference MRI NIfTI file.
    ref_roi_path : str
        Path to the ROI NIfTI file.
    B0_slice_index : int
        Index of the B0 slice to use for scaling.
    save_path : str
        Path to save the scaled MRI NIfTI file.
    """
    # Load config
    ivim_cfg = CONFIG["ivim"]
    snr = ivim_cfg["snr"]
    target_std = 1 / snr  # desired standard deviation

    # Load MRI and ROI
    ref_mri_img = nib.load(ref_mri_path)
    ref_mri = ref_mri_img.get_fdata()
    affine = ref_mri_img.affine
    header = ref_mri_img.header

    ref_roi = nib.load(ref_roi_path).get_fdata().astype(int)

    print(f"ref_mri.shape = {ref_mri.shape}") # should be (x,y,z,b)
    print(f"ref_roi.shape = {ref_roi.shape}")

    # Ensure ROI has the same dimensions as MRI
    if ref_roi.ndim == 3:
        extended_roi = np.zeros_like(ref_mri)
        extended_roi[:, :, :, B0_slice_index] = ref_roi
        ref_roi = extended_roi

    # Validations
    roi_sum = np.sum(ref_roi[:, :, :, B0_slice_index])
    total_sum = np.sum(ref_roi)
    if roi_sum != total_sum:
        raise ValueError("All segmentations must be on the B0 slice.")

    if ref_mri.shape != ref_roi.shape:
        raise ValueError(f"MRI shape {ref_mri.shape} and ROI shape {ref_roi.shape} must match.")

    if len(np.unique(ref_roi)) != 2:
        raise ValueError(f"ROI should be binary. Found values: {np.unique(ref_roi)}")

    # Compute scaling factor
    std_ref = np.std(ref_mri[ref_roi != 0])
    print(f"Reference ROI std = {std_ref:.4f}")
    
    scale_factor = target_std / std_ref
    print(f"Scaling MRI by factor = {scale_factor:.4f}")

    scaled_mri = ref_mri * scale_factor

    # Warn if values exceed 1
    high_voxel_count = np.sum(scaled_mri[ref_roi != 0] >= 1)
    if high_voxel_count > 0:
        # depending on this value SNR can be increased in config.py...
        print(f"Warning: {high_voxel_count} voxel(s) exceed 1 after scaling.")

    # Save scaled MRI
    scaled_img = nib.Nifti1Image(scaled_mri, affine, header)
    nib.save(scaled_img, save_path)
    print(f"Scaled MRI saved to {save_path}")


if __name__ == "__main__":
    scale_invivo_by_snr(
        ref_mri_path="in_vivo_image.nii.gz",
        ref_roi_path="homogeneous_ROI.nii.gz",
        B0_slice_index=0,
        save_path="scaled_in_vivo_image.nii.gz"
    )