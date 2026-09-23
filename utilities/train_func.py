import numpy as np
from scipy import ndimage
from skimage import segmentation as skimage_seg
from scipy.ndimage import distance_transform_edt as distance


def compute_distance_map(labels):
    """
    Input: labels [N, C, H, W, D]
    Output: distance2boundary [NHWD, 1]
    """
    labels = labels.cpu().detach().numpy()
    if labels.ndim == 5:
        N, C, H, W, D = labels.shape
        distance2boundary_batch = np.zeros((N, H, W, D))
        
        for n in range(N):
            boundary = np.zeros((H, W, D))
            for c in range(1, C):
                img_gt = labels[n, c, :, :, :]
                posmask = img_gt > 0
                boundary += skimage_seg.find_boundaries(posmask, connectivity=img_gt.ndim, mode='thick').astype(np.uint16)
            boundary_bool = boundary > 0
            distance2boundary_batch[n] = distance(~boundary_bool) # [N, H, W, D]
            distance2boundary = distance2boundary_batch.reshape(-1, 1) # [NHWD, 1]
    elif labels.ndim == 4:
        N, C, H, W = labels.shape
        distance2boundary_batch = np.zeros((N, H, W))
        
        for n in range(N):
            boundary = np.zeros((H, W))
            for c in range(1, C):
                img_gt = labels[n, c, :, :]
                posmask = img_gt > 0
                boundary += skimage_seg.find_boundaries(posmask, connectivity=img_gt.ndim, mode='thick').astype(np.uint16)
            boundary_bool = boundary > 0
            distance2boundary_batch[n] = distance(~boundary_bool) # [N, H, W]
            distance2boundary = distance2boundary_batch.reshape(-1, 1) # [NHW, 1]
    
    return distance2boundary

def pad_volume_to_patch(volume, patch_size):
    D, H, W = volume.shape
    pd, ph, pw = patch_size

    pad_d = max(0, pd - D)
    pad_h = max(0, ph - H)
    pad_w = max(0, pw - W)

    pad_info = (
        0, pad_d,
        0, pad_h,
        0, pad_w
    )

    padded = np.pad(
        volume,
        ((0, pad_d), (0, pad_h), (0, pad_w)),
        mode="constant",
        constant_values=volume.min()
    )

    return padded, pad_info

def keep_lcc_for_selected_classes(
    label_map: np.ndarray,
    num_classes: int,
    lcc_classes=None,
    connectivity: int = 1):

    output = label_map.copy()
    structure = ndimage.generate_binary_structure(3, connectivity)

    # determine target classes
    if lcc_classes is None:
        target_classes = range(1, num_classes)
    else:
        target_classes = lcc_classes

    for cls in target_classes:
        if cls <= 0 or cls >= num_classes:
            continue

        mask = (label_map == cls)
        if mask.sum() == 0:
            continue

        labeled, num_cc = ndimage.label(mask, structure=structure)
        if num_cc <= 1:
            continue

        sizes = ndimage.sum(mask, labeled, index=range(1, num_cc + 1))
        largest_cc = int(np.argmax(sizes) + 1)

        # remove all but the largest component
        output[mask] = 0
        output[labeled == largest_cc] = cls

    return output

def unpad_2d(x, pad_info):
    if pad_info is None:
        return x

    _, _, ph1, ph2, pw1, pw2 = pad_info

    if x.ndim == 2:
        return x[ph1 : x.shape[0] - ph2,
                 pw1 : x.shape[1] - pw2]
    elif x.ndim == 3:
        return x[:,
                 ph1 : x.shape[1] - ph2,
                 pw1 : x.shape[2] - pw2]
    else:
        raise ValueError(f"Unsupported ndim={x.ndim}")

def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg
