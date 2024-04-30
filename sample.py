from functools import partial
import os
import argparse
import yaml
import torch
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from util.logger import get_logger
import cv2
import numpy as np
from skimage.io import imsave
import warnings
from guided_diffusion.config_manager import ConfigManager
import nibabel as nib

warnings.filterwarnings('ignore')

def save_image_correctly(image_array, output_path):
    """Ensure images are saved correctly by converting to uint8 if necessary."""
    if image_array.dtype != np.uint8:
        # Normalize and convert to uint8
        image_array = 255 * ((image_array - image_array.min()) / (image_array.max() - image_array.min()))
        image_array = image_array.astype(np.uint8)

    # Save using skimage.io.imsave
    imsave(output_path, image_array)

def save_as_nifti(output_folder, output_nifti_path):
    # List all files in the output directory
    files = sorted([f for f in os.listdir(output_folder) if f.endswith('.jpeg') or f.endswith('.png')])
    
    # Load the first image to get dimensions
    example_img = Image.open(os.path.join(output_folder, files[0]))
    example_array = np.array(example_img)
    
    # Initialize an empty array with the shape of [height, width, number of slices]
    all_slices = np.empty((example_array.shape[0], example_array.shape[1], len(files)), dtype=example_array.dtype)
    
    # Iterate over files and stack them
    for i, file_name in enumerate(files):
        img = Image.open(os.path.join(output_folder, file_name))
        all_slices[:, :, i] = np.array(img)
    
    # Create a NIfTI image (default to affine identity matrix)
    nifti_img = nib.Nifti1Image(all_slices, affine=np.eye(4))
    
    # Save as .nii.gz
    nib.save(nifti_img, output_nifti_path)


from functools import partial
import os
import argparse
import yaml
import torch
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from util.logger import get_logger
import cv2
import numpy as np
from skimage.io import imsave
import warnings
from guided_diffusion.config_manager import ConfigManager

warnings.filterwarnings('ignore')

def save_image_correctly(image_array, output_path):
    """Ensure images are saved correctly by converting to uint8 if necessary."""
    if image_array.dtype != np.uint8:
        # Normalize and convert to uint8
        image_array = 255 * ((image_array - image_array.min()) / (image_array.max() - image_array.min()))
        image_array = image_array.astype(np.uint8)

    # Save using skimage.io.imsave
    imsave(output_path, image_array)

def image_read(path, mode='GRAY'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':  
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def load_and_process_images(base_folder, patient_id, modality_suffix):
    img_path = os.path.join(base_folder, patient_id + modality_suffix)
    images = []
    for slice_filename in os.listdir(img_path):
        slice_path = os.path.join(img_path, slice_filename)
        img = cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, axis=0)  # Add channel dimension
        img = img / 255.0 * 2 - 1  # Normalize to [-1, 1]
        images.append((img, slice_filename))
    return images

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_config', type=str, default='configs/model_config_imagenet.yaml')
#     parser.add_argument('--diffusion_config', type=str, default='configs/diffusion_config.yaml')
#     parser.add_argument('--gpu', type=int, default=0)
#     parser.add_argument('--method', choices=['GEM', 'smooth'], required=True)
#     parser.add_argument('--save_dir', type=str, default='./output')
#     args = parser.parse_args()

#     logger = get_logger()
#     device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu')
#     logger.info(f"Device set to {device}")

#     config = ConfigManager.getInstance()
#     config.set_method(args.method)
#     model_config = load_yaml(args.model_config)
#     diffusion_config = load_yaml(args.diffusion_config)
#     model = create_model(**model_config).to(device).eval()
#     sampler = create_sampler(**diffusion_config)
#     sample_fn = partial(sampler.p_sample_loop, model=model)

#     ir_base_folder = 'input/ir'
#     vi_base_folder = 'input/vi'
#     out_path = os.path.join(args.save_dir, args.method.lower())
#     os.makedirs(out_path, exist_ok=True)
    
#     for img_dir in ['recon', 'progress']:
#         os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

#     i = 0
#     for patient_id in os.listdir(ir_base_folder):
#         adc_images = load_and_process_images(ir_base_folder, patient_id, '')
#         zmap_images = load_and_process_images(vi_base_folder, patient_id, '_0001')
        
#         for (adc_img, adc_name), (zmap_img, zmap_name) in zip(adc_images, zmap_images):
#             adc_tensor = torch.tensor(adc_img, dtype=torch.float32).unsqueeze(0).to(device)
#             zmap_tensor = torch.tensor(zmap_img, dtype=torch.float32).unsqueeze(0).to(device)

#             logger.info(f"Inference for image {i}")
#             with torch.no_grad():
#                 sample = sample_fn(x_start=torch.randn(adc_tensor.shape, device=device), record=True, I=adc_tensor, V=zmap_tensor, save_root=out_path, img_index=patient_id, lamb=0.5, rho=0.001)
            
#             sample = sample.cpu().squeeze().numpy()
#             sample = np.transpose(sample, (1, 2, 0))
#             sample = cv2.cvtColor(sample, cv2.COLOR_RGB2YCrCb)[:, :, 0]
#             sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample)) * 255
#             sample_image_path = os.path.join(out_path, 'recon', f"{patient_id}_{adc_name}")
#             imsave(sample_image_path, sample)
#             i += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, default='configs/model_config_imagenet.yaml')
    parser.add_argument('--diffusion_config', type=str, default='configs/diffusion_config.yaml')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--method', choices=['GEM', 'smooth'], required=True)
    parser.add_argument('--save_dir', type=str, default='./output')
    args = parser.parse_args()

    logger = get_logger()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device set to {device}")

    config = ConfigManager.getInstance()
    config.set_method(args.method)
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    model = create_model(**model_config).to(device).eval()
    sampler = create_sampler(**diffusion_config)
    sample_fn = partial(sampler.p_sample_loop, model=model)

    ir_base_folder = 'input/ir'
    vi_base_folder = 'input/vi'
    out_path = os.path.join(args.save_dir, args.method.lower())
    os.makedirs(out_path, exist_ok=True)
    
    for img_dir in ['recon', 'progress']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    i = 0
    for patient_id in os.listdir(ir_base_folder):
        adc_images = load_and_process_images(ir_base_folder, patient_id, '')
        zmap_images = load_and_process_images(vi_base_folder, patient_id, '_0001')
        patient_slices = []

        
        for (adc_img, adc_name), (zmap_img, zmap_name) in zip(adc_images, zmap_images):
            adc_tensor = torch.tensor(adc_img, dtype=torch.float32).unsqueeze(0).to(device)  # Ensure it's 4D
            zmap_tensor = torch.tensor(zmap_img, dtype=torch.float32).unsqueeze(0).to(device)  # Ensure it's 4D

            x_start = torch.randn((adc_tensor.repeat(1, 3, 1, 1)).shape, device=device)  

            with torch.no_grad():

                sample = sample_fn(x_start=x_start, record=True, I=adc_tensor, V=zmap_tensor, save_root=out_path, img_index=patient_id, lamb=0.5, rho=0.001)

            sample = sample.cpu().squeeze().numpy()  # Ensure it's 2D
            patient_slices.append(sample)

        # Convert list of slices into a 3D numpy array
        patient_volume = np.stack(patient_slices, axis=0)
        
        # Create a new NIfTI image from this array
        new_img = nib.Nifti1Image(patient_volume, affine=np.eye(4))
        nifti_save_path = os.path.join(out_path, f"{patient_id}_0002.nii.gz")
        nib.save(new_img, nifti_save_path)

        logger.info(f"Saved NIfTI file for {patient_id} at {nifti_save_path}")