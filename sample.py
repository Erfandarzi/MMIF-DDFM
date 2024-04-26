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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str,default = 'configs/model_config_imagenet.yaml')
    parser.add_argument('--diffusion_config', type=str,default='configs/diffusion_config.yaml')                     
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--method', choices=['GEM', 'smooth'], required=True, help='Specify the method to use (GEM or smooth)')
    parser.add_argument('--save_dir', type=str, default='./output')
    args = parser.parse_args()
    config = ConfigManager.getInstance()
    config.set_method(args.method)

    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    model_config = load_yaml(args.model_config)  
    diffusion_config = load_yaml(args.diffusion_config)
   
    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

  
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop, model=model)
   
    # Working directory
    test_folder=r"input"     
    # Adjust output path based on the method
    out_path = os.path.join(args.save_dir, args.method.lower())
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['recon', 'progress']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    i=0
    for img_name in os.listdir(os.path.join(test_folder,"ir")):
        inf_img = image_read(os.path.join(test_folder,"vi",img_name),mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0 
        vis_img = image_read(os.path.join(test_folder,"ir",img_name), mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0 

        inf_img = inf_img*2-1
        vis_img = vis_img*2-1

        # crop to make divisible
        scale = 32
        h, w = inf_img.shape[2:]
        h = h - h % scale
        w = w - w % scale

        inf_img = ((torch.FloatTensor(inf_img))[:,:,:h,:w]).to(device)
        vis_img = ((torch.FloatTensor(vis_img))[:,:,:h,:w]).to(device)
        assert inf_img.shape == vis_img.shape

        logger.info(f"Inference for image {i}")

        # Sampling
        seed = 3407
        torch.manual_seed(seed)
        x_start = torch.randn((inf_img.repeat(1, 3, 1, 1)).shape, device=device)  

        with torch.no_grad():
            sample = sample_fn(x_start=x_start, record=True, I = inf_img, V = vis_img, save_root=out_path, img_index = os.path.splitext(img_name)[0], lamb=0.5,rho=0.001)

        sample= sample.detach().cpu().squeeze().numpy()
        sample=np.transpose(sample, (1,2,0))
        sample=cv2.cvtColor(sample,cv2.COLOR_RGB2YCrCb)[:,:,0]
        sample=(sample-np.min(sample))/(np.max(sample)-np.min(sample))
        sample=((sample)*255)
        # imsave(os.path.join(os.path.join(out_path, 'recon'), "{}.jpeg".format(img_name.split(".")[0])),sample)
        save_image_correctly(sample, os.path.join(os.path.join(out_path, 'recon'), f"{img_name.split('.')[0]}.jpeg"))

        i = i+1
