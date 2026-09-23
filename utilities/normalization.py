import torch
import numpy as np


def minmax_norm_for_vis(x, eps=1e-6):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x_min = x.min()
    x_max = x.max()
    if x_max > x_min:
        return (x - x_min) / (x_max - x_min + eps)
    else:
        return np.zeros_like(x)
    
def percentile_zscore_norm(images, stats=None):
    norm_images = images.clone()

    def _norm_single(img, stats):
        if stats is not None:
            low, high, mean, std = stats
            img = torch.clamp(img, low, high)
            img = (img - mean) / (std + 1e-8)
        else:
            # flatten for percentile computation
            flat = img.reshape(-1)
            low = torch.quantile(flat, 0.005)
            high = torch.quantile(flat, 0.995)
            img = torch.clamp(img, low, high)
            img = (img - img.mean()) / (img.std(unbiased=False) + 1e-8)
        
        return img
    
    if norm_images.ndim == 3:
        norm_images = _norm_single(norm_images, stats)
    elif norm_images.ndim >= 4:
        for i in range(norm_images.shape[0]):
            norm_images[i, ...] = _norm_single(norm_images[i, ...], stats)

    return norm_images
