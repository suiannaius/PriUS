import torch
import random
from utilities.normalization import percentile_zscore_norm


def generate_noisy_images(images, device, mu_set=0.0, mu_range=None, sigma_set=None, sigma_range=(0, 1.0), intensity_set=1.0, intensity_range=None, seed=None, z_score=False, data_stats=None):
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
        else:
            assert NotImplementedError

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
        noisy_images = percentile_zscore_norm(noisy_images, stats=data_stats)
    # else:
    #     noisy_images = max_min_norm(noisy_images)
    return noisy_images, mu, sigma, intensity

def sample_class_wise_noised_whole_images(images, device, upper_bound=50, seed=None, sigma_1=None, sigma_2=None, int_flag=False, z_score=False, data_stats=None, **kwargs):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    images = images.clone().to(device)

    if kwargs.get('intensity') is None:
        intensity1, intensity2 = 1, 1
    else:
        intensity1, intensity2 = kwargs.get('intensity'), kwargs.get('intensity')
    mu1, mu2 = 0., 0.

    if int_flag:
        sigma1_range = list(range(0, upper_bound)) 
        sigma2_range = list(range(0, upper_bound)) 
    else:
        sigma1_range = (0, upper_bound)
        sigma2_range = (0, upper_bound)
        
    noised_images_1, sampled_mu1, sampled_sigma1, sampled_intensity1 = generate_noisy_images(images, device, mu_set=mu1, sigma_range=sigma1_range, intensity_set=intensity1, seed=seed, sigma_set=sigma_1, z_score=z_score, data_stats=data_stats)
    noised_images_2, sampled_mu2, sampled_sigma2, sampled_intensity2 = generate_noisy_images(images, device, mu_set=mu2, sigma_range=sigma2_range, intensity_set=intensity2, seed=seed, sigma_set=sigma_2, z_score=z_score, data_stats=data_stats)

    sampled_results = {'mu1': sampled_mu1,
                       'mu2': sampled_mu2,
                       'sigma1': sampled_sigma1,
                       'sigma2': sampled_sigma2,
                       'intensity1': sampled_intensity1,
                       'intensity2': sampled_intensity2
                       }

    return sampled_results, noised_images_1, noised_images_2