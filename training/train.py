import os
import json
import torch
import torch.nn.functional as F
from training.criterions import nu_loss_d, nu_loss_far, nu_loss_sigma, gradient_loss
from training.metrics import calculate_dice
from utilities.color import apply_color_map, apply_heatmap
from utilities.utils import safe_item
from utilities.noise import sample_class_wise_noised_whole_images
from utilities.gradient import generate_blurred_images, compute_gradient
from utilities.train_func import l2_regularisation, compute_distance_map
from utilities.normalization import minmax_norm_for_vis, percentile_zscore_norm
from training.criterions import edl_loss, Dice_CE_Loss
from model.models import Uentropy, model_PU


def train_whs(model,
              dataloader,
              optimizer,
              args,
              current_epoch, 
              device,
              writer,
              use_grad_clip=False,
              sample_size=None,
              data_stats=None,
              **kwargs):   
    model.train()

    running_loss = 0.0
    running_dice_loss = 0.0
    running_dice_LV = 0.0
    running_dice_RV = 0.0
    running_dice_LA = 0.0
    running_dice_RA = 0.0
    running_dice_Myo = 0.0
    running_dice_AO = 0.0
    running_dice_PA = 0.0
    if args.method in ['Ours', 'DEviS', 'PureEvidential']:
        running_kl_loss = 0.0
        running_cu_loss = 0.0
    if args.method == 'Ours':
        running_grad_loss = 0.0
        running_noise_loss = 0.0
        running_noise_loss_d = 0.0
        running_noise_loss_sigma = 0.0
        running_noise_loss_far = 0.0
    
    C = args.num_classes
    
    for batch_idx, (images_ori, labels, _, img_paths) in enumerate(dataloader):
        images_ori, labels = images_ori.to(device), labels.to(device) # [N,M,D,H,W], [N,C,D,H,W]
        img_norm = percentile_zscore_norm(images_ori, stats=data_stats)
        if args.use_noise_aug:
            sampled_results, noised_images_1, noised_images_2 = sample_class_wise_noised_whole_images(images_ori, device, upper_bound=args.sigma_upper_bound, z_score=True, intensity=args.intensity, data_stats=data_stats)
            img_norm = torch.concat([img_norm, noised_images_1, noised_images_2], dim=0)
            labels = torch.concat([labels, labels, labels], dim=0)
        hard_label = torch.argmax(labels, dim=1) # [N,D,H,W]
        targets = labels.permute(0, 2, 3, 4, 1).contiguous().view(-1, C) # [NDHW, C]
        N, _, D, H, W = img_norm.size()
        optimizer.zero_grad()

        if args.method in ['PlainUNet', 'EU', 'UDrop', 'TTA']:  # Plain UNet methods
            logits = model(img_norm)
            prob = F.softmax(logits, dim=1).permute(0, 2, 3, 4, 1).contiguous().view(-1, C) # [NDHW,C]
            dice_loss = Dice_CE_Loss(targets,
                                     prob,
                                     logits,
                                     hard_label,
                                     epsilon=1e-5,
                                     device=device,
                                     num_classes=C)
            loss = torch.mean(dice_loss)
        
        elif args.method == 'PU':  # Probabilistic UNet
            onehot_target = labels.float()
            model.forward(img_norm, onehot_target, training=True)
            elbo = model.elbo(onehot_target)
            reg_loss = l2_regularisation(model.posterior) + l2_regularisation(model.prior) + l2_regularisation(model.fcomb.layers)
            loss = -elbo + 1e-5 * reg_loss
            with torch.no_grad():
                model.forward(img_norm, training=False)
                logits = model_PU(img_norm, model)
                prob = F.softmax(logits, dim=1).permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
                dice_loss = Dice_CE_Loss(targets,
                                         prob,
                                         logits,
                                         hard_label,
                                         epsilon=1e-5,
                                         device=device,
                                         num_classes=C)
        
        if args.method in ['PlainUNet', 'PU', 'UDrop', 'TTA', 'EU']:
            with torch.no_grad():
                u = Uentropy(logits.detach(), C).view(-1, 1)
                u_view = u.view(N, D, H, W) # [N,D,H,W]
        
        if args.method in ['Ours', 'DEviS', 'PureEvidential']:  # Evidential methods
            if args.use_prior:
                json_path="./label_statistics.json"
                with open(json_path, "r") as f:
                    stats = json.load(f)
                assert stats["center"] == args.center_filter, f"Wrong center stats."
                ratios_dict = stats[args.modality_filter]["class_ratios"]
                base_rate_list = [ratios_dict[str(i)] for i in range(len(ratios_dict))]
                base_rate = torch.tensor(base_rate_list, dtype=torch.float32, device=device).unsqueeze(0)
                prior = base_rate * args.num_classes
                if batch_idx == 0 and current_epoch == 0:
                    print('Using class-ratio based prior...')
                    print(f'Prior: {prior.detach().cpu()}')
            else:
                prior = torch.ones((1, args.num_classes), dtype=torch.float32, device=device)

            logits = model(img_norm)
            evidence = F.softplus(logits).permute(0, 2, 3, 4, 1).contiguous().view(-1, C) # [NDHW,C]
            alpha = evidence + prior
            S = torch.sum(alpha, dim=1, keepdim=True) # [NDHW,1]
            u = args.num_classes / S # [NDHW,1]
            u_view = u.view(N, D, H, W) # [N,D,H,W]
            prob = alpha / S # [NDHW,C]
            evi_loss, _, kl_loss, dice_loss, cu_loss = edl_loss(targets, 
                                                                alpha, 
                                                                num_classes=args.num_classes, 
                                                                current_epoch=current_epoch, 
                                                                total_epoch=args.num_epochs,
                                                                annealing_steps=args.annealing_steps, 
                                                                device=device, 
                                                                loss_type=args.loss_type,
                                                                prior=prior,
                                                                coef_kl=args.coef_kl,
                                                                coef_cu=args.coef_cu,
                                                                **kwargs)
            if args.method != 'Ours':
                loss = evi_loss
            else:  # Uncertainty Supervision
                # Initialize losses
                noise_loss_sigma = torch.zeros(1, device=device)
                noise_loss_d = torch.zeros(1, device=device)
                noise_loss_far = torch.zeros(1, device=device)
                grad_loss =torch.zeros(1, device=device)
                noise_loss =torch.zeros(1, device=device)
                distance_map = compute_distance_map(labels).reshape(N, D, H, W)
                
                # Noise loss
                if args.gamma != 0 and current_epoch >= args.good_model_step:
                    frozen_ratio = 0.9
                    if current_epoch <= int(frozen_ratio * args.num_epochs):
                        upper_bound = int(args.sigma_upper_bound * (current_epoch - args.good_model_step) / (int(frozen_ratio * args.num_epochs) - args.good_model_step))
                    else:
                        upper_bound = args.sigma_upper_bound
                    if batch_idx == 0:
                        print(f'Current Epoch: {current_epoch}, Upper bound for noise sampling: {upper_bound}')
                    sampled_results, noised_images_1, noised_images_2 = sample_class_wise_noised_whole_images(images_ori,
                                                                                                              device, 
                                                                                                              upper_bound=upper_bound,
                                                                                                              z_score=True,
                                                                                                              intensity=args.intensity,
                                                                                                              data_stats=data_stats)
                    evidence_noised_1 = F.softplus(model(noised_images_1).permute(0, 2, 3, 4, 1).contiguous().view(-1, C))
                    evidence_noised_2 = F.softplus(model(noised_images_2).permute(0, 2, 3, 4, 1).contiguous().view(-1, C))
                    alpha_1 = evidence_noised_1 + 1
                    alpha_2 = evidence_noised_2 + 1
                    S_1 = torch.sum(alpha_1, dim=1, keepdim=True) # [NDHW,1]
                    S_2 = torch.sum(alpha_2, dim=1, keepdim=True) # [NDHW,1]
                    uncertainty_noised_1 = C / S_1  # [NDHW,1]
                    uncertainty_noised_2 = C / S_2
                    del evidence_noised_1, evidence_noised_2
                    if args.coef_sigma != 0.0:
                        noise_loss_sigma_1 = args.coef_sigma * nu_loss_sigma(uncertainty_noised_1.view(N, D, H, W, 1).permute(0, 4, 1, 2, 3).clone().detach(), 
                                                                             uncertainty_noised_2.view(N, D, H, W, 1).permute(0, 4, 1, 2, 3), 
                                                                             sampled_results['sigma1'],
                                                                             sampled_results['sigma2'],
                                                                             distance_map,
                                                                             device=device,
                                                                             threshold=args.threshold_sigma)
                        
                        noise_loss_sigma_2 = args.coef_sigma * nu_loss_sigma(u.view(N, D, H, W, 1).permute(0, 4, 1, 2, 3).clone().detach(), 
                                                                             uncertainty_noised_1.view(N, D, H, W, 1).permute(0, 4, 1, 2, 3), 
                                                                             0,
                                                                             sampled_results['sigma1'],
                                                                             distance_map,
                                                                             device=device,
                                                                             threshold=args.threshold_sigma)

                        noise_loss_sigma = noise_loss_sigma_1 + noise_loss_sigma_2
                    if args.coef_d != 0.0:
                        noise_loss_d = args.coef_d * (nu_loss_d(u.view(N, D, H, W, 1).permute(0, 4, 1, 2, 3),
                                                                distance_map,
                                                                labels,
                                                                device=device,
                                                                threshold=args.threshold_d) + 
                                                      nu_loss_d(uncertainty_noised_1.view(N, D, H, W, 1).permute(0, 4, 1, 2, 3),
                                                                distance_map,
                                                                labels,
                                                                device=device,
                                                                threshold=args.threshold_d) + 
                                                      nu_loss_d(uncertainty_noised_2.view(N, D, H, W, 1).permute(0, 4, 1, 2, 3),
                                                                distance_map,
                                                                labels,
                                                                device=device,
                                                                threshold=args.threshold_d))           
                    if args.coef_far != 0.0:
                        noise_loss_far = args.coef_far * nu_loss_far(u.view(N, D, H, W, 1).permute(0, 4, 1, 2, 3),
                                                                     uncertainty_noised_1.view(N, D, H, W, 1).permute(0, 4, 1, 2, 3),
                                                                     uncertainty_noised_2.view(N, D, H, W, 1).permute(0, 4, 1, 2, 3),
                                                                     distance_map,
                                                                     device=device,
                                                                     threshold=args.threshold_far)
                    noise_loss = noise_loss_sigma + noise_loss_d + noise_loss_far
                    del alpha_1, alpha_2, S_1, S_2
                
                if args.beta != 0 and current_epoch >= args.good_model_step:
                    blurred_images = generate_blurred_images(img_norm, sigma_blur=args.sigma_blur, device=device)  
                    gradient_blurred = compute_gradient(blurred_images) # [NDHW,1]
                    grad_loss = gradient_loss(u, gradient_blurred, distance_map, args.batch_size, device=device, sample_size=sample_size, onehot_label=labels, num_classes=args.num_classes)
                
                annealing_start = torch.tensor(0.01, dtype=torch.float32)
                annealing_AU = annealing_start * torch.exp(-torch.log(annealing_start) / (args.num_epochs - args.good_model_step) * (current_epoch - args.good_model_step))
                loss = evi_loss + (args.gamma * noise_loss + args.beta * grad_loss) * annealing_AU
        loss.backward()
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        dice_LV, dice_RV, dice_LA, dice_RA, dice_Myo, dice_AO, dice_PA = calculate_dice(targets, prob, epsilon=1e-5, device=device, num_classes=C)
        
        running_loss += safe_item(loss)
        running_dice_loss += safe_item(torch.mean(dice_loss))
        running_dice_LV += safe_item(dice_LV)
        running_dice_RV += safe_item(dice_RV)
        running_dice_LA += safe_item(dice_LA)
        running_dice_RA += safe_item(dice_RA)
        running_dice_Myo += safe_item(dice_Myo)
        running_dice_AO += safe_item(dice_AO)
        running_dice_PA += safe_item(dice_PA)
        
        if args.method in ['Ours', 'DEviS', 'PureEvidential']:
            running_kl_loss += safe_item(kl_loss)
            running_cu_loss += safe_item(cu_loss)
        if args.method == 'Ours':
            running_noise_loss += safe_item(noise_loss)
            running_grad_loss += safe_item(grad_loss)
            running_noise_loss_d += safe_item(noise_loss_d)
            running_noise_loss_sigma += safe_item(noise_loss_sigma)
            running_noise_loss_far += safe_item(noise_loss_far)
        
        # Visualization  
        if writer is not None and batch_idx == 0:
            writer.add_scalar('(Train) L0-1. Loss', running_loss / len(dataloader), current_epoch)
            writer.add_scalar('(Train) L0-2. Loss Dice', running_dice_loss / len(dataloader), current_epoch)
            
            if args.method in ['Ours', 'DEviS', 'PureEvidential']:
                writer.add_scalar('(Train) L0-3. Loss KL', running_kl_loss / len(dataloader), current_epoch)
                writer.add_scalar('(Train) L0-4. Loss CU', running_cu_loss / len(dataloader), current_epoch)
                if args.method == 'Ours':
                    if args.gamma != 0 and current_epoch >= args.good_model_step:
                        writer.add_scalar('(Train) L1. Loss Noise (no annealing)', noise_loss, current_epoch)
                        if args.coef_sigma != 0:
                            writer.add_scalar('(Train) L2. Loss sigma (no annealing) (sigma1-2)', noise_loss_sigma_1/args.coef_sigma, current_epoch)
                            writer.add_scalar('(Train) L3. Loss sigma (no annealing) (0-sigma1)', noise_loss_sigma_2/args.coef_sigma, current_epoch)
                        if args.coef_d != 0:
                            writer.add_scalar('(Train) L4. Loss d (no annealing)', noise_loss_d/args.coef_d, current_epoch)
                        if args.coef_far != 0:
                            writer.add_scalar('(Train) L5.  Loss far (no annealing)', noise_loss_far/args.coef_far, current_epoch)
                    if args.beta != 0 and current_epoch >= args.good_model_step:
                        writer.add_scalar('(Train) L0. Loss Grad (no annealing)', grad_loss, current_epoch)
            
            prob_view = prob.view(N, D, H, W, C).permute(0, 4, 1, 2, 3)
            prob_indices = torch.argmax(prob_view, dim=1) # [N,D,H,W]

            for i in range(len(img_paths)):
                if hard_label[i].sum() == 0:
                    continue
                fg_slices = torch.where(hard_label[i].sum(dim=(1, 2)) > 0)[0]
                if len(fg_slices) == 0:
                    continue
                d_slice = fg_slices[len(fg_slices) // 2].item()

                prob_slice = prob_indices[i, d_slice, :, :].detach().cpu().numpy() # [H,W]
                prob_color = apply_color_map(prob_slice)
                prob_color = prob_color.transpose(2, 0, 1)
                
                label_slice = hard_label[i, d_slice, :, :].detach().cpu().numpy()  # [H,W]               
                label_color = apply_color_map(label_slice) # [H,W,3]
                label_color = label_color.transpose(2, 0, 1)  # [3,H,W]

                u_view = u.view(N, D, H, W, 1).squeeze(-1) # [N,H,W]
                u_slice = u_view[i, d_slice, :, :].detach().cpu().numpy() # [H,W]
                u_color = apply_heatmap(u_slice) # [H,W,3]
                u_color = u_color.transpose(2, 0, 1)  # [3,H,W]

                file_name = os.path.basename(img_paths[i])
                writer.add_image(f'(Train) A1. Ground Truth/{file_name}', label_color, dataformats='CHW')  # label_color.astype(np.uint8)
                writer.add_image(f'(Train) A2. Predicted Mask/{file_name}', prob_color, global_step=current_epoch, dataformats='CHW')
                writer.add_image(f'(Train) A3. Original Image/{file_name}', minmax_norm_for_vis(img_norm[i, :, d_slice, :, :]), dataformats='CHW')
                writer.add_image(f'(Train) B1. Uncertainty_Original/{file_name}', u_color, global_step=current_epoch, dataformats='CHW')
                del prob_slice, label_slice, u_slice, prob_color, label_color, u_color

                if args.method == 'Ours':
                    if args.gamma != 0 and current_epoch >= args.good_model_step:
                        u_view_1 = uncertainty_noised_1.view(N, D, H, W) # [N,D,H,W]
                        u_slice_1 = u_view_1[i, d_slice, :, :].detach().cpu().numpy()  # [H,W]
                        u_color_1 = apply_heatmap(u_slice_1) # [H,W,3]
                        u_color_1 = u_color_1.transpose(2, 0, 1)  # [3,H,W]
                        writer.add_image(f"(Train) B2. Uncertainty1/{file_name}", u_color_1, global_step=current_epoch, dataformats='CHW')

                        u_view_2 = uncertainty_noised_2.view(N, D, H, W) # [N,D, H,W]
                        u_slice_2 = u_view_2[i, d_slice, :, :].detach().cpu().numpy()  # [H,W]
                        u_color_2 = apply_heatmap(u_slice_2) # [H,W,3]
                        u_color_2 = u_color_2.transpose(2, 0, 1)  # [3,H,W]
                        writer.add_image(f"(Train) B3. Uncertainty2/{file_name}", u_color_2, global_step=current_epoch, dataformats='CHW')
                        
                        writer.add_image(f'(Train) A4. Noised Image1/{file_name}', minmax_norm_for_vis(noised_images_1[i, :, d_slice, :, :]), global_step=current_epoch, dataformats='CHW')
                        writer.add_image(f'(Train) A5. Noised Image2/{file_name}', minmax_norm_for_vis(noised_images_2[i, :, d_slice, :, :]), global_step=current_epoch, dataformats='CHW')
                        
                        writer.add_scalar(f'(Train) sigma1', sampled_results['sigma1'], global_step=current_epoch)
                        writer.add_scalar(f'(Train) sigma2', sampled_results['sigma2'], global_step=current_epoch)
                    
                    if args.beta != 0 and current_epoch >= args.good_model_step:
                        grad_blurred_slice = gradient_blurred.view(N, D, H, W, 1).permute(0, 4, 1, 2, 3)[i, :, d_slice, :, :]
                        writer.add_image(f'(Train) Z. Gradient_blurred/{file_name}', grad_blurred_slice.squeeze(0), dataformats='HW')
        
        del img_norm, labels, hard_label
        del dice_LV, dice_RV, dice_LA, dice_RA, dice_Myo, dice_AO, dice_PA
        
    return running_loss / len(dataloader), running_dice_LV / len(dataloader), running_dice_RV / len(dataloader), running_dice_LA / len(dataloader), \
        running_dice_RA / len(dataloader), running_dice_Myo / len(dataloader), running_dice_AO / len(dataloader), running_dice_PA / len(dataloader)
        