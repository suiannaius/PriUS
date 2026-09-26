import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


def compute_distance_map_numpy(lbl):
    import scipy.ndimage as ndi

    dist = np.zeros_like(lbl, dtype=np.float32)
    for c in np.unique(lbl):
        if c == 0:
            continue
        mask = lbl == c
        dist_c = ndi.distance_transform_edt(mask)
        dist[mask] = dist_c[mask]
    return dist


def compute_gradient_numpy(image, sigma_blur=0.5):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    img_norm = max_min_norm(image)
    blurred_images = generate_blurred_images(img_norm, sigma_blur=sigma_blur)
    assert blurred_images.ndim == 2
    gradients = np.gradient(blurred_images, axis=(-2, -1))
    value = np.sqrt(gradients[0] ** 2 + gradients[1] ** 2) + 1e-8

    return value


def max_min_norm(images):
    norm_images = images.copy()
    if norm_images.ndim == 3:
        norm_images = (norm_images - norm_images.min()) / max(
            float(norm_images.max() - norm_images.min()), 1e-12
        )
    elif norm_images.ndim >= 4:
        for i in range(norm_images.shape[0]):
            norm_images[i, ...] = (
                norm_images[i, ...] - norm_images[i, ...].min()
            ) / max(float(norm_images[i, ...].max() - norm_images[i, ...].min()), 1e-12)
    return norm_images


def generate_blurred_images(images, sigma_blur=0.5):
    if sigma_blur == 0.0:
        return images
    else:
        blurred_images = np.zeros_like(images)
        for i in range(images.shape[0]):
            blurred_images[i] = gaussian_filter(images[i], sigma=sigma_blur)
        return blurred_images


def aggregate_along_normal(u, g, kernel_size=3):
    """
    u: [N,1,D,H,W] uncertainty
    g: [N,1,D,H,W] gradient magnitude
    """
    pad = kernel_size // 2

    g_agg = F.max_pool2d(g, kernel_size, stride=1, padding=pad)  # 用窗口近似法线
    u_agg = F.avg_pool2d(u, kernel_size, stride=1, padding=pad)

    return u_agg, g_agg


def sigmoid_indicator(x, alpha=10.0):
    return torch.sigmoid(alpha * x)


def penalty(x, type="identity"):
    if type == "identity":
        return x
    elif type == "exp":
        return torch.exp(x)
