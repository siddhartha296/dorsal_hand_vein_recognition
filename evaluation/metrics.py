"""
Performance metrics for GAN quality and Authentication accuracy
"""
import torch
import numpy as np
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

class VeinMetrics:
    def __init__(self, device='cuda'):
        self.device = device

    @staticmethod
    def calculate_ssim_psnr(real_images, fake_images):
        """Calculates average SSIM and PSNR between real and synthetic batches"""
        ssim_vals = []
        psnr_vals = []
        
        # Convert to numpy and scale to [0, 1] for skimage
        real_np = ((real_images.cpu().numpy() + 1) / 2).squeeze()
        fake_np = ((fake_images.cpu().numpy() + 1) / 2).squeeze()

        for i in range(real_np.shape[0]):
            ssim_vals.append(ssim(real_np[i], fake_np[i], data_range=1.0))
            psnr_vals.append(psnr(real_np[i], fake_np[i], data_range=1.0))
            
        return np.mean(ssim_vals), np.mean(psnr_vals)

    @staticmethod
    def calculate_eer(y_true, y_scores):
        """Calculates Equal Error Rate (EER) for authentication"""
        fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
        fnr = 1 - tpr
        
        # EER is where FPR == FNR
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        thresh = interp1d(fpr, thresholds)(eer)
        
        return eer, thresh

    def calculate_fid(self, real_features, fake_features):
        """
        Calculates Frechet Inception Distance
        Note: Requires feature vectors from a pre-trained inception/classifier model
        """
        mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
        
        ssdiff = np.sum((mu1 - mu2)**2.0)
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid