import torch
import numpy as np
import random


def max_min_norm(images):
    if not isinstance(images, torch.Tensor):
        images = torch.from_numpy(images)
    norm_images = images.clone()
    if norm_images.ndim == 3:
        norm_images = (norm_images - norm_images.min()) / (
            norm_images.max() - norm_images.min()
        ).clamp_min(1e-12)
    elif norm_images.ndim >= 4:
        for i in range(norm_images.shape[0]):
            norm_images[i, ...] = (norm_images[i, ...] - norm_images[i, ...].min()) / (
                norm_images[i, ...].max() - norm_images[i, ...].min()
            ).clamp_min(1e-12)
    return norm_images


def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):
    for param_group in optimizer.param_groups:
        param_group["lr"] = round(init_lr * np.power(1 - (epoch) / max_epoch, power), 8)


def generate_noisy_images(
    images,
    device,
    mu_set=0.0,
    mu_range=None,
    sigma_set=None,
    sigma_range=(0, 1.0),
    intensity_set=1.0,
    intensity_range=None,
    seed=None,
    z_score=False,
):
    images = images.clone().to(device)
    if seed is not None:
        torch.manual_seed(seed)

    if mu_set is not None:
        mu = mu_set
    else:
        assert mu_range is not None, f"mu_range is None"
        mu = random.uniform(*mu_range)

    if sigma_set is not None:
        sigma = sigma_set
    else:
        if isinstance(sigma_range, tuple) and len(sigma_range) == 2:
            sigma = random.uniform(*sigma_range)
        elif isinstance(sigma_range, list) and len(sigma_range) > 0:
            sigma = random.choice(sigma_range)

    if intensity_set is not None:
        intensity = intensity_set
    else:
        assert intensity_range is not None, f"intensity_range is None"
        intensity = random.uniform(*intensity_range)
    if images.ndim == 4:
        _, _, H, W = images.shape
        noise = torch.normal(mean=mu, std=sigma, size=(H, W)).to(device)
    elif images.ndim == 5:
        _, _, H, W, D = images.shape
        noise = torch.normal(mean=mu, std=sigma, size=(H, W, D)).to(device)

    noisy_images = images + intensity * noise
    if z_score:
        noisy_images = percentile_zscore_norm(noisy_images)
    else:
        noisy_images = max_min_norm(noisy_images)
    return noisy_images, mu, sigma, intensity


def sample_class_wise_noised_whole_images(
    images,
    device,
    upper_bound=50,
    seed=None,
    sigma_1=None,
    sigma_2=None,
    int_flag=False,
    z_score=False,
    **kwargs,
):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    images = images.clone().to(device)

    if kwargs.get("intensity") is None:
        intensity1, intensity2 = 1, 1
    else:
        intensity1, intensity2 = kwargs.get("intensity"), kwargs.get("intensity")
    mu1, mu2 = 0.0, 0.0

    if int_flag:
        sigma1_range = list(range(1, upper_bound))
        sigma2_range = list(range(1, upper_bound))
    else:
        sigma1_range = (0, upper_bound)
        sigma2_range = (0, upper_bound)

    noised_images_1, sampled_mu1, sampled_sigma1, sampled_intensity1 = (
        generate_noisy_images(
            images,
            device,
            mu_set=mu1,
            sigma_range=sigma1_range,
            intensity_set=intensity1,
            seed=seed,
            sigma_set=sigma_1,
            z_score=z_score,
        )
    )
    noised_images_2, sampled_mu2, sampled_sigma2, sampled_intensity2 = (
        generate_noisy_images(
            images,
            device,
            mu_set=mu2,
            sigma_range=sigma2_range,
            intensity_set=intensity2,
            seed=seed,
            sigma_set=sigma_2,
            z_score=z_score,
        )
    )

    sampled_results = {
        "mu1": sampled_mu1,
        "mu2": sampled_mu2,
        "sigma1": sampled_sigma1,
        "sigma2": sampled_sigma2,
        "intensity1": sampled_intensity1,
        "intensity2": sampled_intensity2,
    }

    return sampled_results, noised_images_1, noised_images_2


def percentile_zscore_norm(images):
    """
    images: torch.Tensor
        shape = (C, H, W) or (N, C, H, W, ...)
    return:
        normalized torch.Tensor, same shape as input
    """
    norm_images = images.clone()

    def _norm_single(img):
        # flatten for percentile computation
        flat = img.reshape(-1)
        low = torch.quantile(flat, 0.005)
        high = torch.quantile(flat, 0.995)

        img = torch.clamp(img, low, high)
        img = (img - img.mean()) / (img.std(unbiased=False) + 1e-8)
        return img

    if norm_images.ndim == 3:
        norm_images = _norm_single(norm_images)

    elif norm_images.ndim >= 4:
        for i in range(norm_images.shape[0]):
            norm_images[i, ...] = _norm_single(norm_images[i, ...])

    return norm_images
