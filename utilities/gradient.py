import torch
import numpy as np
from scipy.ndimage import gaussian_filter, sobel


def generate_blurred_images(images, device, sigma_blur=0.5):
    if sigma_blur==0.0:
        return images
    else:
        images = images.clone().to(device)
        blurred_images = torch.zeros_like(images)
        for i in range(images.shape[0]):
            blurred_images[i] = torch.from_numpy(gaussian_filter(images[i].cpu().numpy(), sigma=sigma_blur))
        return blurred_images

def compute_gradient(image):
    image = image.detach().cpu().numpy()
    
    if image.ndim == 5:  # 5D tensor [N, C, H, W, D]
        n, c, h, w, d = image.shape

        if c > 1:
            image_flat = image.reshape(n * c, h, w, d)
            gradients = np.gradient(image_flat, axis=(-3, -2, -1))
            magnitude = np.sqrt(gradients[0]**2 + gradients[1]**2 + gradients[2]**2) + 1e-8
            magnitude = magnitude.reshape(n, c, h, w, d)

            value = np.max(magnitude, axis=1, keepdims=True)  # [N, 1, H, W, D]
        else:
            gradients = np.gradient(image, axis=(-3, -2, -1))
            value = np.sqrt(gradients[0]**2 + gradients[1]**2 + gradients[2]**2) + 1e-8
        
        value = np.transpose(value, (0, 2, 3, 4, 1)).reshape(-1, 1)  # [NHWD, 1]
        return torch.from_numpy(value)
    
    elif image.ndim == 4:  # 4D tensor [N, C, H, W]
        n, c, h, w = image.shape

        value = compute_sobel_gradient(image) * 4  # [N, 1, H, W]
        value = np.transpose(value, (0, 2, 3, 1)).reshape(-1, 1)  # [NHW, 1]
        return torch.from_numpy(value)

def compute_sobel_gradient(image_np):
    N, C, H, W = image_np.shape

    # 如果是RGB图像，转换为灰度
    if C > 1:
        image_np = np.mean(image_np, axis=1)  # [N, H, W]
    else:
        image_np = image_np[:, 0, :, :]  # [N, H, W]

    gradients = []
    for i in range(N):
        gx = sobel(image_np[i], axis=1, mode='reflect')  # dx
        gy = sobel(image_np[i], axis=0, mode='reflect')  # dy
        grad = np.sqrt(gx ** 2 + gy ** 2)
        gradients.append(grad)

    gradients = np.stack(gradients)  # [N, H, W]
    gradients = gradients[:, np.newaxis, :, :]  # [N, 1, H, W]
    return gradients
