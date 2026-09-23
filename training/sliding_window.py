import torch
import torch.nn.functional as F
from model.models import model_PU
from utilities.color import apply_color_map
from utilities.normalization import minmax_norm_for_vis


def axis_coords(L, p, s):
    """Generate patch start coordinates along one axis, ensuring last patch is included."""
    coords = list(range(0, L - p + 1, s))
    if len(coords) == 0:
        coords = [0]
    elif coords[-1] != L - p:
        coords.append(L - p)
    return coords

def get_gaussian_weight(patch_size, sigma_scale=0.6):
    pd, ph, pw = patch_size
    zz, yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, pd),
        torch.linspace(-1, 1, ph),
        torch.linspace(-1, 1, pw),
        indexing="ij")
    
    dist = zz**2 + yy**2 + xx**2
    sigma = sigma_scale
    weight = torch.exp(-dist / (2 * sigma ** 2))

    weight = weight / weight.max()
    return weight.unsqueeze(0).unsqueeze(0)  # [1,1,pd,ph,pw]

def tta_transforms_3d():
    transforms = []

    # identity
    transforms.append((
        lambda x: x,
        lambda x: x
    ))

    # flip H
    transforms.append((
        lambda x: torch.flip(x, dims=[3]),
        lambda x: torch.flip(x, dims=[3])
    ))

    # flip W
    transforms.append((
        lambda x: torch.flip(x, dims=[4]),
        lambda x: torch.flip(x, dims=[4])
    ))

    # flip H + W
    transforms.append((
        lambda x: torch.flip(x, dims=[3, 4]),
        lambda x: torch.flip(x, dims=[3, 4])
    ))

    # rotate 90 in HW plane
    transforms.append((
        lambda x: torch.rot90(x, k=1, dims=[3, 4]),
        lambda x: torch.rot90(x, k=3, dims=[3, 4])
    ))

    # rotate + flip H
    transforms.append((
        lambda x: torch.flip(torch.rot90(x, 1, [3, 4]), dims=[3]),
        lambda x: torch.rot90(torch.flip(x, dims=[3]), 3, [3, 4])
    ))

    # rotate + flip W
    transforms.append((
        lambda x: torch.flip(torch.rot90(x, 1, [3, 4]), dims=[4]),
        lambda x: torch.rot90(torch.flip(x, dims=[4]), 3, [3, 4])
    ))

    # rotate + flip H + W
    transforms.append((
        lambda x: torch.flip(torch.rot90(x, 1, [3, 4]), dims=[3, 4]),
        lambda x: torch.rot90(torch.flip(x, dims=[3, 4]), 3, [3, 4])
    ))

    return transforms

def sliding_window_inference(
    volume,
    label,
    model,
    patch_size,
    stride,
    num_classes,
    device,
    writer=None,
    global_step=0,
    tag_prefix="SW",
    method=None,
    **kwargs):

    _, _, D, H, W = volume.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    # Pad if volume is smaller than patch size
    pad_d = max(0, pd - D)
    pad_h = max(0, ph - H)
    pad_w = max(0, pw - W)

    pd1, pd2 = pad_d // 2, pad_d - pad_d // 2
    ph1, ph2 = pad_h // 2, pad_h - pad_h // 2
    pw1, pw2 = pad_w // 2, pad_w - pad_w // 2

    volume = F.pad(volume, (pw1, pw2, ph1, ph2, pd1, pd2), "constant", volume.min())
    label = F.pad(label.unsqueeze(1).float(), (pw1, pw2, ph1, ph2, pd1, pd2), "constant", 0).long().squeeze(1)

    _, _, Dp, Hp, Wp = volume.shape

    output = torch.zeros((1, num_classes, Dp, Hp, Wp), device=device)
    count_map = torch.zeros((1, 1, Dp, Hp, Wp), device=device)

    zs = axis_coords(Dp, pd, sd)
    ys = axis_coords(Hp, ph, sh)
    xs = axis_coords(Wp, pw, sw)

    patch_counter = 0
    center_z = Dp // 2

    sliding_mode = kwargs.get("sliding_mode", "gaussian")
    if sliding_mode == 'avg':
        # gaussian_weight = 1
        gaussian_weight = torch.ones((1, 1, pd, ph, pw), device=device)
    elif sliding_mode == 'gaussian':
        gaussian_weight = get_gaussian_weight(patch_size).to(device)

    tta_ops = tta_transforms_3d() if method == 'TTA' else None
    
    for z in zs:
        for y in ys:
            for x in xs:
                img_patch = volume[:, :, z:z+pd, y:y+ph, x:x+pw]
                label_patch = label[:, z:z+pd, y:y+ph, x:x+pw]
                
                if method in ['Ours', 'DEviS', 'PureEvidential', 'PlainUNet', 'UDrop']:
                    logits = model(img_patch)
                    if method in ['Ours', 'DEviS', 'PureEvidential']:
                        evidence = F.softplus(logits)  # [1,C,pd,ph,pw]
                        prior = kwargs.get("prior", None)
                        if prior is None:
                            alpha = evidence + 1.0
                        else:
                            # prior: [1, C] → [1, C, 1, 1, 1]
                            prior = prior.view(1, -1, 1, 1, 1).to(logits.device)
                            alpha = evidence + prior
                        S = torch.sum(alpha, dim=1, keepdim=True) # [1,1,pd,ph,pw]
                        prob = alpha / S  # [1,C,pd,ph,pw]

                    elif method in ['PlainUNet', 'UDrop']:
                        prob = F.softmax(logits, dim=1)
                
                elif method == 'PU':
                    logits = model_PU(img_patch, model).unsqueeze(0)  # [1,C,pd,ph,pw]
                    prob = F.softmax(logits, dim=1)
                
                elif method == 'TTA':
                    logits_sum = 0.0
                    for aug, deaug in tta_ops:
                        aug_patch = aug(img_patch)
                        aug_logits = model(aug_patch)
                        logits_sum += deaug(aug_logits)
                    logits = logits_sum / len(tta_ops)
                    prob = F.softmax(logits, dim=1)
                
                elif method == 'EU':
                    num_models = len(model)
                    logits = 0
                    for i in range(num_models):
                        model[i].eval()
                        logits += model[i](img_patch)
                    logits /= num_models
                    prob = F.softmax(logits, dim=1)
                
                else:
                    raise NotImplementedError(f'Method {method} not implemented.')
                
                if writer is not None and (z <= center_z < z + pd):
                    local_z = center_z - z
                    img_slice = img_patch[0, 0, local_z].detach().cpu()
                    prob_indices = torch.argmax(prob, dim=1).detach().cpu().numpy() # [1,pd,ph,pw]
                    prob_slice = prob_indices[0, local_z, :, :]  # [1,ph,pw]
                    prob_color = apply_color_map(prob_slice).transpose(2, 0, 1)  # [3,ph,pw]
                    gt_slice = label_patch[0, local_z, :, :].detach().cpu().numpy()  # [1,ph,pw]
                    gt_color = apply_color_map(gt_slice).transpose(2, 0, 1)   # [3,ph,pw]
                    
                    writer.add_image(
                        f"{tag_prefix}/patch_{patch_counter}_A_image",
                        minmax_norm_for_vis(img_slice.unsqueeze(0)),  # [1,H,W]
                        global_step,
                        dataformats="CHW")
                    writer.add_image(
                        f"{tag_prefix}/patch_{patch_counter}_B_gt",
                        gt_color,
                        global_step,
                        dataformats="CHW")
                    writer.add_image(
                        f"{tag_prefix}/patch_{patch_counter}_C_pred",
                        prob_color,
                        global_step,
                        dataformats="CHW")
 
                    patch_counter += 1
                output[:, :, z:z+pd, y:y+ph, x:x+pw] += logits * gaussian_weight
                count_map[:, :, z:z+pd, y:y+ph, x:x+pw] += gaussian_weight

    output = output / count_map.clamp_min(1e-8)
    return output  # [1,C,Dp,Hp,Wp]
