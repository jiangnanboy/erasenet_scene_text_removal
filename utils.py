from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np

def AdjustLearningRate(optimizer, lr):
    for param_group in optimizer.param_groups:
        print('param_group',param_group['lr'])
        param_group['lr'] = lr

def compute_psnr(im1, im2):
    p = psnr(im1, im2)
    return p

def compute_ssim(im1, im2):
    isRGB = len(im1.shape) == 3 and im1.shape[-1] == 3
    s = ssim(im1, im2, K1=0.01, K2=0.03, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
             multichannel=isRGB)
    return s

def pd_tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    img = tensor.squeeze().cpu().numpy()
    img = img.clip(min_max[0], min_max[1])
    img = (img - min_max[0]) / (min_max[1] - min_max[0])
    if out_type == np.uint8:
        # scaling
        img = img * 255.0
    img = np.transpose(img, (1, 2, 0))
    img = img.round()
    img = img[:,:,::-1]
    return img.astype(out_type)

