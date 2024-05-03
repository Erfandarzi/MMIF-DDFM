import nibabel as nib
from PIL import Image
import numpy as np
import os

def nii_to_jpeg_per_channel(nii_file_path, output_folder_adc, output_folder_zmap):
    # Load the NIfTI file
    img = nib.load(nii_file_path)
    data = img.get_fdata()
    
    # Check if there are at least two channels
    if data.shape[-1] < 2:
        raise ValueError("The NIfTI file does not have two channels")

    # Separate the channels
    adc_data = data[..., 0]  # Assuming first channel is ADC
    zmap_data = data[..., 1]  # Assuming second channel is Zmap

    # Process each channel
    process_slices(adc_data, output_folder_adc, "ADC")
    process_slices(zmap_data, output_folder_zmap, "Zmap")

def process_slices(channel_data, output_folder, modality):
    # Check if output directory exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all slices along the third dimension (assumes channel data is at the end)
    for i, slice in enumerate(channel_data.transpose(2, 0, 1)):  # Adjust depending on how your data is oriented
        # Normalize and convert to uint8
        slice_image = ((slice - np.min(slice)) / (np.max(slice) - np.min(slice)) * 255).astype(np.uint8)
        # Convert to PIL Image and convert to grayscale
        pil_image = Image.fromarray(slice_image).convert('L')
        # Save the image slice as JPEG
        pil_image.save(os.path.join(output_folder, f'{modality}_slice_{i:04d}.jpeg'))

# # Specify the directory containing NIfTI files
# input_dir = './SaM-Med3D_HQ/data/train/brain_lesion/Task501_HIE/imagesTr'
# adc_output_dir = './MMIF-DDFM/input/ir'  # Output directory for ADC slices
# zmap_output_dir = './MMIF-DDFM/input/vi'  # Output directory for Zmap slices

# # Process all NIfTI files in the directory
# for filename in os.listdir(input_dir):
#     if filename.endswith(".nii.gz"):  # Check for NIfTI files
#         file_path = os.path.join(input_dir, filename)
#         nii_to_jpeg_per_channel(file_path, adc_output_dir, zmap_output_dir)

import nibabel as nib
import numpy as np

def inspect_nifti(file_path):
    # Load the NIfTI file
    img = nib.load(file_path)
    data = img.get_fdata()

    # Print basic file information
    print("NIfTI Header:", img.header)
    print("Data shape:", data.shape)
    print("Data type:", data.dtype)

    # Calculate and display basic statistics
    print("Mean intensity value:", np.mean(data))
    print("Standard deviation:", np.std(data))
    print("Max intensity value:", np.max(data))
    print("Min intensity value:", np.min(data))

    # Check for NaN values
    if np.isnan(data).any():
        print("Data contains NaN values.")
    else:
        print("No NaN values in the data.")
# Specify the path to your NIfTI file
# nifti_file_path = './output/gem/HIE_313_0002.nii.gz'

def check_slice_consistency(nifti_path):
    img = nib.load(nifti_path)
    data = img.get_fdata()

    # Example: check each slice for variations in max and min values
    for i in range(data.shape[-1]):  # Assuming the last dimension is the slice dimension
        slice_data = data[..., i]
        print(f"Slice {i} - Max: {np.max(slice_data)}, Min: {np.min(slice_data)}")


nifti_file_path = "../SaM-Med3D_HQ/data/train/brain_lesion/Task501_HIE/imagesTr/HIE_379.nii.gz"
inspect_nifti(nifti_file_path)
check_slice_consistency(nifti_file_path)

nifti_file_path = "./output/gem/HIE_379_0002.nii.gz"
inspect_nifti(nifti_file_path)
check_slice_consistency(nifti_file_path)
