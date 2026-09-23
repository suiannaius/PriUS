import os
import torch
import torch.nn.functional as F
import matplotlib.cm as cm
import json
from tqdm import tqdm
from training.metrics import soft_hd95, calculate_dice, calculate_dice_from_labels
from training.sliding_window import sliding_window_inference
from model.models import Uentropy
from utilities.color import apply_color_map, apply_heatmap
from utilities.noise import sample_class_wise_noised_whole_images
from utilities.train_func import compute_distance_map, unpad_2d, keep_lcc_for_selected_classes
from utilities.normalization import minmax_norm_for_vis, percentile_zscore_norm
from utilities.gradient import compute_gradient, generate_blurred_images
from utilities.count_pixels import count_pixels_d_chunk, count_pixels_sigma, count_pixels_grad


def inference_WHS(model, dataloader, device, writer, args, consider_spacing=False, data_stats=None, **kwargs):
    if args.method != 'EU':
        model.eval()
    running_dice_LV, running_dice_RV, running_dice_LA, running_dice_RA, running_dice_Myo, running_dice_AO, running_dice_PA = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    running_hd95_LV, running_hd95_RV, running_hd95_LA, running_hd95_RA, running_hd95_Myo, running_hd95_AO, running_hd95_PA = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    total_ratio_sigma = 0.0
    total_corr_sigma = 0.0
    total_corr_d = 0.0
    total_ratio_d = 0.0
    total_corr_g = 0.0
    total_ratio_grad = 0.0
    
    args.batch_size = 1
    C = args.num_classes
    total_samples = len(dataloader.dataset)
    print(f"Using Device: {device}")
    print('Inference in progress...')
    
    if args.method in ['Ours', 'DEviS', 'PureEvidential']:
        print('Using Evidential Methods...')
        # ----------- Use prior -----------
        if args.use_prior:
            json_path="./label_statistics.json"
            with open(json_path, "r") as f:
                stats = json.load(f)
            ratios_dict = stats[args.modality_filter]["class_ratios"]
            base_rate_list = [ratios_dict[str(i)] for i in range(len(ratios_dict))]
            base_rate = torch.tensor(base_rate_list, dtype=torch.float32, device=device).unsqueeze(0)
            prior = base_rate * args.num_classes
        else:
            prior = torch.ones((1, args.num_classes), dtype=torch.float32, device=device)
    else:
        prior = None
    
    with torch.no_grad(), tqdm(total=total_samples) as progress_bar:
        for batch_idx, (images_ori, labels, spacing, img_paths, pad_info) in enumerate(dataloader):
            progress_bar.set_description(f"Case: {os.path.basename(img_paths[0])}")
            images_ori, labels = images_ori.to(device), labels.to(device) # [N,M,D,H,W], [N,C,D,H,W]
            img_norm = percentile_zscore_norm(images_ori, stats=data_stats)
            hard_label = torch.argmax(labels, dim=1) # [N,D,H,W]
            targets = labels.permute(0, 2, 3, 4, 1).contiguous().view(-1, C) # [NDHW, C]
            logits = sliding_window_inference(img_norm,
                                              hard_label,
                                              model,
                                              patch_size=args.patch_size,
                                              stride=tuple(s // 2 for s in args.patch_size),
                                              num_classes=C,
                                              device=device,
                                              writer=writer,
                                              global_step=batch_idx,
                                              tag_prefix=f"WHS/{os.path.basename(img_paths[0])}",
                                              prior=prior,
                                              method=args.method,
                                              **kwargs)  # [N,C,Dp,Hp,Wp]
            N, _, D, H, W = logits.size()

            if args.method in ['Ours', 'DEviS', 'PureEvidential']:  # In this case, logits are evidences.
                logits = logits.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
                alpha = F.softplus(logits) + prior
                S = torch.sum(alpha, dim=1, keepdim=True) # [NDpHpWp,1]
                u = args.num_classes / S # [NDpHpWp,1]
                u_view = u.view(N, D, H, W) # [N,Dp,Hp,Wp]
                prob = alpha / S # [NDpHpWp,C]
            
            elif args.method in ['PlainUNet', 'UDrop', 'TTA', 'EU', 'PU']:
                prob = F.softmax(logits, dim=1).permute(0, 2, 3, 4, 1).contiguous().view(-1, C) # [NDHW,C]
                u = Uentropy(logits.detach(), C).view(-1, 1)
                u_view = u.view(N, D, H, W) # [N,D,H,W]
                
            else:
                raise NotImplementedError(f'Method {args.method} not implemented.')

            prob_view = prob.view(N, D, H, W, C).permute(0, 4, 1, 2, 3) # [N,C,D,H,W]
            prob_indices = torch.argmax(prob_view, dim=1) # [N,D,H,W]
            
            use_lcc = kwargs.get("use_lcc", False)
            if use_lcc:
                lcc_classes = kwargs.get("lcc_classes", None)
                prob_indices_pp = []
                for n in range(prob_indices.shape[0]):
                    pred_np = prob_indices[n].detach().cpu().numpy()
                    pred_np = keep_lcc_for_selected_classes(
                        label_map=pred_np,
                        num_classes=C,
                        lcc_classes=lcc_classes,
                        connectivity=1)
                    prob_indices_pp.append(torch.from_numpy(pred_np))

                prob_indices = torch.stack(prob_indices_pp, dim=0).to(device)

            if use_lcc:
                dice = calculate_dice_from_labels(pred_labels=prob_indices, gt_labels=hard_label, num_classes=C, epsilon=1e-5, device=device)
            else:
                dice = calculate_dice(targets, prob, epsilon=1e-5, device=device, num_classes=C)
        
            if consider_spacing:
                hd95 = soft_hd95(hard_label, prob_indices, spacing, num_classes=C)
            else:
                hd95 = soft_hd95(hard_label, prob_indices, num_classes=C)
            
            dice_LV, dice_RV, dice_LA, dice_RA, dice_Myo, dice_AO, dice_PA = dice[0], dice[1], dice[2], dice[3], dice[4], dice[5], dice[6]
            hd95_LV, hd95_RV, hd95_LA, hd95_RA, hd95_Myo, hd95_AO, hd95_PA = hd95[0], hd95[1], hd95[2], hd95[3], hd95[4], hd95[5], hd95[6]

            distance_map = compute_distance_map(labels).reshape(N, D, H, W)
            
            # ----------- Generate Noised Images and Inference Again -----------
            intensity = getattr(args, "intensity", 1)
            sampled_results, noised_images_1, noised_images_2 = sample_class_wise_noised_whole_images(images_ori,
                                                                                                      device=device,
                                                                                                      seed=args.seed, 
                                                                                                      sigma_1=10, 
                                                                                                      sigma_2=args.sigma_upper_bound,
                                                                                                      z_score=True,
                                                                                                      intensity=intensity,
                                                                                                      data_stats=data_stats)
            logits_1 = sliding_window_inference(noised_images_1,
                                                hard_label,
                                                model,
                                                patch_size=args.patch_size,
                                                stride=tuple(s // 2 for s in args.patch_size),
                                                num_classes=C,
                                                device=device,
                                                writer=None,
                                                global_step=batch_idx,
                                                tag_prefix=f"WHS/{os.path.basename(img_paths[0])}",
                                                prior=prior,
                                                method=args.method,
                                                **kwargs)
            
            logits_2 = sliding_window_inference(noised_images_2,
                                                hard_label,
                                                model,
                                                patch_size=args.patch_size,
                                                stride=tuple(s // 2 for s in args.patch_size),
                                                num_classes=C,
                                                device=device,
                                                writer=None,
                                                global_step=batch_idx,
                                                tag_prefix=f"WHS/{os.path.basename(img_paths[0])}",
                                                prior=prior,
                                                method=args.method,
                                                **kwargs)
            
            if args.method in ['Ours', 'DEviS', 'PureEvidential']:  # In this case, logits are evidences.
                logits_1 = logits_1.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
                alpha_1 = F.softplus(logits_1) + prior
                S_1 = torch.sum(alpha_1, dim=1, keepdim=True) # [NDHW,1]
                u_1 = args.num_classes / S_1 # [NDHW,1]
                u_1_view = u_1.view(N, D, H, W) # [N,D,H,W]
                prob_1 = alpha_1 / S_1 # [NDHW,C]
                
                logits_2 = F.softplus(logits_2).permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
                alpha_2 = logits_2 + prior
                S_2 = torch.sum(alpha_2, dim=1, keepdim=True) # [NDHW,1]
                u_2 = args.num_classes / S_2 # [NDHW,1]
                u_2_view = u_2.view(N, D, H, W) # [N,D,H,W]
                prob_2 = alpha_2 / S_2 # [NDHW,C]

            elif args.method in ['PlainUNet', 'UDrop', 'TTA', 'EU', 'PU']:  # In this case, 'logits are logits'.
                prob_1 = F.softmax(logits_1, dim=1).permute(0, 2, 3, 4, 1).contiguous().view(-1, C) # [NDHW,C]
                u_1 = Uentropy(logits_1.detach(), C).view(-1, 1)
                u_1_view = u_1.view(N, D, H, W) # [N,D,H,W]
                
                prob_2 = F.softmax(logits_2, dim=1).permute(0, 2, 3, 4, 1).contiguous().view(-1, C) # [NDHW,C]
                u_2 = Uentropy(logits_2.detach(), C).view(-1, 1)
                u_2_view = u_2.view(N, D, H, W) # [N,D,H,W]
            
            prob_1_view = prob_1.view(N, D, H, W, C).permute(0, 4, 1, 2, 3) # [N,C,D,H,W]
            prob_2_view = prob_2.view(N, D, H, W, C).permute(0, 4, 1, 2, 3) # [N,C,D,H,W]
            prob_1_indices = torch.argmax(prob_1_view, dim=1) # [N,D,H,W]
            prob_2_indices = torch.argmax(prob_2_view, dim=1) # [N,D,H,W]

            if use_lcc:
                lcc_classes = kwargs.get("lcc_classes", None)
                prob_1_indices_pp = []
                prob_2_indices_pp = []
                for n in range(prob_indices.shape[0]):
                    pred_1_np = prob_1_indices[n].detach().cpu().numpy()
                    pred_1_np = keep_lcc_for_selected_classes(
                        label_map=pred_1_np,
                        num_classes=C,
                        lcc_classes=lcc_classes,
                        connectivity=1)
                    prob_1_indices_pp.append(torch.from_numpy(pred_1_np))
                    
                    pred_2_np = prob_2_indices[n].detach().cpu().numpy()
                    pred_2_np = keep_lcc_for_selected_classes(
                        label_map=pred_2_np,
                        num_classes=C,
                        lcc_classes=lcc_classes,
                        connectivity=1)
                    prob_2_indices_pp.append(torch.from_numpy(pred_2_np))

                prob_1_indices = torch.stack(prob_1_indices_pp, dim=0).to(device)
                prob_2_indices = torch.stack(prob_2_indices_pp, dim=0).to(device)

            blurred_images = generate_blurred_images(img_norm, sigma_blur=0.5, device=device)  
            gradient_blurred = compute_gradient(blurred_images) # [NDHW,1]
            gradient_1 = compute_gradient(generate_blurred_images(noised_images_1, sigma_blur=0.5, device=device)) # [NDHW,1]
            gradient_2 = compute_gradient(generate_blurred_images(noised_images_2, sigma_blur=0.5, device=device)) # [NDHW,1]
            writer.add_histogram("Gradient Distribution", gradient_blurred, global_step=batch_idx)

            # Calculate UCC and UR
            ratio_sigma, corr_sigma = count_pixels_sigma(u_1, u_2, sampled_results['sigma1'], sampled_results['sigma2'], distance_map.reshape(-1), device, threshold=args.threshold_sigma)
            ratio_d_0, corr_d_0 = count_pixels_d_chunk(u, distance_map.reshape(-1), labels, batch_size=1, device=device, threshold=args.threshold_d)
            ratio_d_1, corr_d_1 = count_pixels_d_chunk(u_1, distance_map.reshape(-1), labels, batch_size=1, device=device, threshold=args.threshold_d)
            ratio_d_2, corr_d_2 = count_pixels_d_chunk(u_2, distance_map.reshape(-1), labels, batch_size=1, device=device, threshold=args.threshold_d)
            ratio_d, corr_d = (ratio_d_0 + ratio_d_1 + ratio_d_2) / 3., (corr_d_0 + corr_d_1 + corr_d_2) / 3.
            # ratio_d, corr_d = count_pixels_d_chunk(u, distance_map.reshape(-1), labels, batch_size=1, device=device, threshold=args.threshold_d)
            ratio_grad, corr_g = count_pixels_grad(u, gradient_blurred, distance_map.reshape(-1), labels, batch_size=1, device=device, threshold=0)
            
            total_ratio_sigma += ratio_sigma
            total_corr_sigma += corr_sigma
            total_corr_d += corr_d
            total_ratio_d += ratio_d
            total_corr_g += corr_g
            total_ratio_grad += ratio_grad
            
            # ----------- Visualization -----------
            for i in range(img_norm.shape[0]):
                if hard_label[i].sum() == 0:
                    continue
                fg_slices = torch.where(hard_label[i].sum(dim=(1, 2)) > 0)[0]

                if len(fg_slices) == 0:
                    continue
                d_slice = fg_slices[len(fg_slices) // 2].item()
                
                images_slice = img_norm[i, :, d_slice, :, :].cpu().numpy() # [M,H,W]
                img_1_slice = noised_images_1[i, :, d_slice, :, :].cpu().numpy() # [M,H,W]
                img_2_slice = noised_images_2[i, :, d_slice, :, :].cpu().numpy() # [M,H,W]
                
                prob_slice = prob_indices[i, d_slice, :, :].detach().cpu().numpy() # [H,W]
                prob_1_slice = prob_1_indices[i, d_slice, :, :].detach().cpu().numpy() # [H,W]
                prob_2_slice = prob_2_indices[i, d_slice, :, :].detach().cpu().numpy() # [H,W]
                label_slice = hard_label[i, d_slice, :, :].detach().cpu().numpy()  # [H,W]
                u_slice = u_view[i, d_slice, :, :].detach().cpu().numpy()  # [H,W]
                u_1_slice = u_1_view[i, d_slice, :, :].detach().cpu().numpy()
                u_2_slice = u_2_view[i, d_slice, :, :].detach().cpu().numpy()
                distance_map_slice = distance_map.reshape(N, D, H, W)[i, d_slice, :, :]

                # Unpad slices
                pad = pad_info[i]
                images_slice = unpad_2d(images_slice, pad)
                img_1_slice = unpad_2d(img_1_slice, pad)
                img_2_slice = unpad_2d(img_2_slice, pad)

                prob_slice   = unpad_2d(prob_slice, pad)
                prob_1_slice   = unpad_2d(prob_1_slice, pad)
                prob_2_slice   = unpad_2d(prob_2_slice, pad)
                label_slice  = unpad_2d(label_slice, pad)
                u_slice      = unpad_2d(u_slice, pad)
                u_1_slice    = unpad_2d(u_1_slice, pad)
                u_2_slice    = unpad_2d(u_2_slice, pad)
                distance_map_slice = unpad_2d(distance_map_slice, pad)
                
                prob_color = apply_color_map(prob_slice) # [H,W,3]
                prob_1_color = apply_color_map(prob_1_slice) # [H,W,3]
                prob_2_color = apply_color_map(prob_2_slice) # [H,W,3]
                prob_color = prob_color.transpose(2, 0, 1)  # [3,H,W]
                prob_1_color = prob_1_color.transpose(2, 0, 1)  # [3,H,W]
                prob_2_color = prob_2_color.transpose(2, 0, 1)  # [3,H,W]
                
                label_color = apply_color_map(label_slice) # [H,W,3]
                label_color = label_color.transpose(2, 0, 1)  # [3,H,W]
                
                u_color_max = 1.0
                u_color = apply_heatmap(u_slice, vmax=u_color_max) # [H,W,3]
                u_color = u_color.transpose(2, 0, 1)  # [3,H,W]
                u_1_color = apply_heatmap(u_1_slice, vmax=u_color_max) # [H,W,3]
                u_1_color = u_1_color.transpose(2, 0, 1)  # [3,H,W]
                u_2_color = apply_heatmap(u_2_slice, vmax=u_color_max) # [H,W,3]
                u_2_color = u_2_color.transpose(2, 0, 1)  # [3,H,W]

                d_color = apply_heatmap(distance_map_slice, 0, 50) # [H,W,3]
                d_color = d_color.transpose(2, 0, 1)  # [3,H,W]
                difference_min_max = [-1.0, 1.0]  
                u_1_dif_color = apply_heatmap(u_1_slice - u_slice, vmin=difference_min_max[0], vmax=difference_min_max[1]) # [H,W,3]
                u_1_dif_color = u_1_dif_color.transpose(2, 0, 1)  # [3,H,W]
                u_2_dif_color = apply_heatmap(u_2_slice - u_slice, vmin=difference_min_max[0], vmax=difference_min_max[1]) # [H,W,3]
                u_2_dif_color = u_2_dif_color.transpose(2, 0, 1)  # [3,H,W]

                grad_blurred_slice = gradient_blurred.view(N, D, H, W, 1).permute(0, 4, 1, 2, 3)[i, :, D//2, :, :] # [1,H,W]
                grad_blurred_slice_color = grad_blurred_slice.squeeze(0)  # [H,W]
                grad_1_blurred_slice = gradient_1.view(N, D, H, W, 1).permute(0, 4, 1, 2, 3)[i, :, D//2, :, :] # [1,H,W]
                grad_2_blurred_slice = gradient_2.view(N, D, H, W, 1).permute(0, 4, 1, 2, 3)[i, :, D//2, :, :] # [1,H,W]

                grad_norm = (grad_blurred_slice_color - grad_blurred_slice_color.min()) / (grad_blurred_slice_color.max() - grad_blurred_slice_color.min() + 1e-8)
                colormap = cm.get_cmap('jet')
                grad_color = colormap(grad_norm.cpu().numpy())[:, :, :3]  # [H, W, 3]
                grad_color_tensor = torch.from_numpy(grad_color).permute(2, 0, 1)

                writer.add_image(f'(Test) A1. Original Image/{os.path.basename(img_paths[0])}', minmax_norm_for_vis(images_slice), dataformats='CHW')
                writer.add_image(f"(Test) A2. Noisy Image1 (sigma={sampled_results['sigma1']:.3g})/{os.path.basename(img_paths[0])}", minmax_norm_for_vis(img_1_slice), dataformats='CHW')
                writer.add_image(f"(Test) A3. Noisy Image2 (sigma={sampled_results['sigma2']:.3g})/{os.path.basename(img_paths[0])}", minmax_norm_for_vis(img_2_slice), dataformats='CHW')
                
                writer.add_image(f'(Test) B0. Ground Truth/{os.path.basename(img_paths[0])}', label_color, dataformats='CHW')
                writer.add_image(f'(Test) B1. Predicted Mask/{os.path.basename(img_paths[0])}', prob_color, dataformats='CHW')
                writer.add_image(f"(Test) B2. Predicted Mask1 (sigma={sampled_results['sigma1']:.3g})/{os.path.basename(img_paths[0])}", prob_1_color, dataformats='CHW')
                writer.add_image(f"(Test) B3. Predicted Mask2 (sigma={sampled_results['sigma2']:.3g})/{os.path.basename(img_paths[0])}", prob_2_color, dataformats='CHW')
                
                writer.add_image(f'(Test) C1. Uncertainty/{os.path.basename(img_paths[0])}', u_color, dataformats='CHW')
                writer.add_image(f'(Test) C2. Uncertainty1/{os.path.basename(img_paths[0])}', u_1_color, dataformats='CHW')
                writer.add_image(f'(Test) C3. Uncertainty2/{os.path.basename(img_paths[0])}', u_2_color, dataformats='CHW')
                
                writer.add_image(f'(Test) D1. Uncertainty sigma1-ori/{os.path.basename(img_paths[0])}', u_1_dif_color, dataformats='CHW')
                writer.add_image(f'(Test) D2. Uncertainty sigma2-ori/{os.path.basename(img_paths[0])}', u_2_dif_color, dataformats='CHW')
                
                writer.add_image(f'(Test) E1. Gradient_blurred_colormap/{os.path.basename(img_paths[0])}', grad_color_tensor, dataformats='CHW')
                writer.add_image(f'(Test) E2. Gradient_blurred/{os.path.basename(img_paths[0])}', grad_blurred_slice.squeeze(0), dataformats='HW')
                writer.add_image(f'(Test) E3. Gradient_1_blurred/{os.path.basename(img_paths[0])}', grad_1_blurred_slice.squeeze(0), dataformats='HW')
                writer.add_image(f'(Test) E4. Gradient_2_blurred/{os.path.basename(img_paths[0])}', grad_2_blurred_slice.squeeze(0), dataformats='HW')

                writer.add_image(f'(Test) F1. Distance map/{os.path.basename(img_paths[0])}', d_color, dataformats='CHW')
                del images_slice, prob_slice, prob_1_slice, prob_2_slice, label_slice, prob_color, prob_1_color, prob_2_color, label_color, u_color, u_1_color, u_2_color
                # torch.cuda.empty_cache()
            
            running_dice_LV += dice_LV.item()
            running_dice_RV += dice_RV.item()
            running_dice_LA+= dice_LA.item()
            running_dice_RA+= dice_RA.item()
            running_dice_Myo+= dice_Myo.item()
            running_dice_AO+= dice_AO.item()
            running_dice_PA+= dice_PA.item()

            running_hd95_LV += hd95_LV.item()
            running_hd95_RV += hd95_RV.item()
            running_hd95_LA+= hd95_LA.item()
            running_hd95_RA+= hd95_RA.item()
            running_hd95_Myo+= hd95_Myo.item()
            running_hd95_AO+= hd95_AO.item()
            running_hd95_PA+= hd95_PA.item()

            progress_bar.update(images_ori.size(0))
    
    return (
        {'dice':[
        running_dice_LV / len(dataloader),
        running_dice_RV / len(dataloader),
        running_dice_LA / len(dataloader), 
        running_dice_RA / len(dataloader),
        running_dice_Myo / len(dataloader),
        running_dice_AO / len(dataloader),
        running_dice_PA / len(dataloader)],
        
        'hd95':[
        running_hd95_LV / len(dataloader),
        running_hd95_RV / len(dataloader),
        running_hd95_LA / len(dataloader),
        running_hd95_RA / len(dataloader),
        running_hd95_Myo / len(dataloader),
        running_hd95_AO / len(dataloader),
        running_hd95_PA / len(dataloader)],
        
        'ucc_sigma': total_corr_sigma / len(dataloader),
        'ur_sigma': total_ratio_sigma / len(dataloader),
        'ucc_d': total_corr_d / len(dataloader),
        'ur_d': total_ratio_d / len(dataloader),
        'ucc_grad': total_corr_g / len(dataloader),
        'ur_grad': total_ratio_grad / len(dataloader)
        })
        