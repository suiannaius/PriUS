import torch
from utilities.utils import ensure_tensor
from monai.metrics import HausdorffDistanceMetric


def Dice(y_true, y_pred, epsilon, device):
    """
    Input tensor:
    y_true: [NHWD, C]
    y_pred: [NHWD, C]
    alpha: [NHWD, C]
    """
    y_true = y_true.to(device) # [NHWD, C]
    y_pred = y_pred.to(device) # [NHWD, C]
    smooth = torch.zeros(y_pred.size(-1), dtype=torch.float32).fill_(0.00001).to(device) # [C, ]
    ones = torch.ones(y_pred.shape).to(device) # [NHWD, C]
    class_mask = y_true + epsilon
    P = y_pred
    P_ = ones - P
    class_mask_ = ones - class_mask
    TP = P * class_mask
    FP = P * class_mask_
    FN = P_ * class_mask

    A = FP.sum(dim=(0)) / ((FP.sum(dim=(0)) + FN.sum(dim=(0))) + smooth)
    A = torch.clamp(A, min=0.2, max=0.8)
    B = 1 - A
    num = torch.sum(TP, dim=(0)).float()
    den = num + A * torch.sum(FP, dim=(0)).float() + B * torch.sum(FN, dim=(0)).float()
    dice = num / (den + smooth) # (C, )

    return dice

def DiceLoss(y_true, y_pred, epsilon, device):
    return 1 - Dice(y_true, y_pred, epsilon, device)

def CE(y_logits, label, epsilon, device):
    return torch.nn.CrossEntropyLoss()(y_logits, label)

def Dice_CE_Loss(y_true, y_pred, y_logits, hard_label, epsilon, device, coef_ce=0.1, num_classes=None):

    dice_loss = DiceLoss(y_true, y_pred, epsilon, device)
    ce_loss = CE(y_logits, hard_label, epsilon, device)
    loss = dice_loss + coef_ce * ce_loss
    return loss

def KL(alp, c, device, beta=None):
    if beta is None: # uniform prior
        beta = torch.ones((1, c), device=device)
        assert torch.all(alp >= 1), "alp needs to be greater than or equal to 1."
    else:
        beta = beta.to(device).view(1, c)
    alp = alp.to(device) # [NHWD,C]
    S_alp = torch.sum(alp, dim=1, keepdim=True) # [NHWD,1]

    S_beta = torch.sum(beta, dim=1, keepdim=True) # [1,1]
    lnB = torch.lgamma(S_alp) - torch.sum(torch.lgamma(alp), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alp)
    dg1 = torch.digamma(alp)
    kl = torch.sum((alp - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni

    return kl

def edl_loss(y_true, alpha, num_classes, current_epoch, total_epoch, annealing_steps, device, loss_type, coef_cu=0.0, prior=None, **kwargs):
    assert loss_type in ['log', 'digamma', 'mse'], f"'loss_type' should be 'log', 'digamma' or 'mse', but got {loss_type}."
    y_true = y_true.to(device) # [NHWD, 4]
    alpha = alpha.to(device) # [NHWD, 4]
    S = torch.sum(alpha, dim=-1, keepdim=True) # [NHWD, 1]
    prob = alpha / S # [NHWD, 4]
    if loss_type == 'log':
        L_ACE = torch.sum(torch.mul(y_true, (torch.log(S) - torch.log(alpha))), dim=-1, keepdim=True) # [NHWD, 1]
    elif loss_type == 'digamma':
        L_ACE = torch.sum(torch.mul(y_true, (torch.digamma(S) - torch.digamma(alpha))), dim=-1, keepdim=True) # [NHWD, 1]
    elif loss_type == 'mse':
        L_err = torch.sum((y_true - prob) ** 2, dim=-1, keepdim=True) # [NHWD, 1]
        L_var = alpha * (S - alpha) / (S * S * (S + 1)) # [NHWD, 4]
        L_ACE = torch.sum(L_err + L_var, dim=-1, keepdim=True) # [NHWD, 1]
    alp = (alpha - 1) * (1 - y_true) + 1 # [NHWD, 4]
    annealing_coef = min(1, current_epoch / annealing_steps)
    L_KL = KL(alp, num_classes, device, beta=prior) # [NHWD, 1]
    L_DICE = DiceLoss(y_true, prob, epsilon=1e-3, device=device) # [C,]
    annealing_start = torch.tensor(0.01, dtype=torch.float32)
    annealing_AU = annealing_start * torch.exp(-torch.log(annealing_start) / total_epoch * current_epoch)  # from 'annealing_start' to '1'
    # AU Loss
    pred_scores, pred_cls = torch.max(alpha / S, 1, keepdim=True)
    uncertainty = num_classes / S
    target = torch.argmax(y_true, dim=1, keepdims=True)
    acc_match = torch.reshape(torch.eq(pred_cls, target).float(), (-1, 1)) # [NHWD, 1]
    acc_uncertain = - pred_scores * torch.log(1 - uncertainty + 1e-5) # [NHWD, 1]
    inacc_certain = - (1 - pred_scores) * torch.log(uncertainty + 1e-5)
    L_AU = annealing_AU * acc_match * acc_uncertain + (1 - annealing_AU) * (1 - acc_match) * inacc_certain
    loss = torch.mean(L_ACE  + kwargs.get("coef_KL", 1.0) * annealing_coef * L_KL + (1 - annealing_AU) * L_DICE + coef_cu * L_AU)
    L_ACE = torch.mean(L_ACE)
    L_KL = torch.mean(L_KL)
    L_DICE = torch.mean(L_DICE)
    L_AU = torch.mean(L_AU)

    return loss, L_ACE, L_KL, L_DICE, L_AU

def Hausdorff_Distance(y_true, y_pred, device):
    y_true = y_true.to(device)
    y_pred = y_pred.to(device)
    hd_95 = HausdorffDistanceMetric(include_background=True, reduction='none', percentile=95)
    hd_95(y_pred, y_true)
    result = hd_95.aggregate()

    return result

def nu_loss_sigma(u_noised_sigma1, 
                  u_noised_sigma2, 
                  sigma1, 
                  sigma2, 
                  distance_map, 
                  device, 
                  threshold=None):
    u_noised_sigma1 = ensure_tensor(u_noised_sigma1, device)
    u_noised_sigma2 = ensure_tensor(u_noised_sigma2, device)
    batch_size = u_noised_sigma1.shape[0]
    delta_uncertainty = (u_noised_sigma2 - u_noised_sigma1).view(batch_size, -1).to(device)
    delta_sigma = sigma2 - sigma1

    d_mask = torch.from_numpy(distance_map <= threshold).view(batch_size, -1).to(device)
    delta_loss = (delta_uncertainty * delta_sigma)
    uncertainty_map = (delta_loss < 0).float().to(device)
    d_mask = torch.from_numpy(distance_map <= threshold).view(batch_size, -1).to(device)
    
    mask_noised = (uncertainty_map * d_mask).view(batch_size, -1).to(device)
    
    if torch.sum(mask_noised) != 0:
        loss = torch.sum(mask_noised * delta_loss) / torch.sum(mask_noised)
    else:
        loss = torch.zeros(1, device=device)

    return -loss

def nu_loss_d(u, d, label, device, k=20, threshold=None):
    u = ensure_tensor(u, device)
    N, C = u.shape[0], u.shape[1]
    num_classes = label.shape[1]

    u = u.view(N, C, -1)  # (N, C, H*W)
    d = torch.from_numpy(d).view(N, -1).to(device)  # (N, H*W)
    label = label.view(N, num_classes, -1)  # (N, num_classes, H*W)
    
    total_loss = 0.0
    valid_terms = 0

    for b in range(N):
        for c in range(C):
            u_flat = u[b, c]  # (H*W,)
            d_flat = d[b]  # (H*W,)
            
            for class_id in range(num_classes):
                class_mask = (label[b, class_id].bool() & (d_flat <= threshold))
                
                idx = torch.nonzero(class_mask).squeeze()
                if idx.numel() < 2:
                    continue

                if idx.numel() > k:
                    perm = torch.randperm(idx.shape[0])[:k]
                    idx = idx[perm]
                
                u_sampled = u_flat[idx]  # (k,)
                d_sampled = d_flat[idx]  # (k,)
                delta_d = torch.triu(d_sampled.unsqueeze(0) - d_sampled.unsqueeze(1), diagonal=1)  # (k, k)
                delta_u = torch.triu(u_sampled.unsqueeze(0) - u_sampled.unsqueeze(1), diagonal=1)  # (k, k)

                pairwise_loss = delta_d * delta_u
                mask = (pairwise_loss > 0).float()
                
                num_pairs = min(idx.numel(), k)
                if mask.sum() > 0:
                    total_loss = total_loss + torch.sum(torch.exp(pairwise_loss * mask)) / (mask.sum() + 1e-6)
                    # total_loss = total_loss + torch.sum(pairwise_loss * mask) / (mask.sum() + 1e-6)
                    valid_terms = valid_terms + num_pairs * (num_pairs - 1) / 2

    if valid_terms == 0:
        return torch.tensor(0.0, device=device)
    return total_loss / valid_terms

def nu_loss_far(u, u_noised_sigma1, u_noised_sigma2, distance_map, device, threshold=None):
    distance_map_tensor = torch.from_numpy(distance_map).to(device).unsqueeze(1)
    d_mask = (distance_map_tensor > threshold).float()
    loss = torch.sum((u + u_noised_sigma1 + u_noised_sigma2) * d_mask) / torch.sum(d_mask)

    return loss


# def nu_loss_far(u, u_noised_sigma1, u_noised_sigma2, distance_map, device, threshold=None):
#     distance_map_tensor = torch.from_numpy(distance_map).to(device).unsqueeze(1)
#     d_mask = (distance_map_tensor > threshold).float()
    
#     loss1 = torch.sum((u + u_noised_sigma1 + u_noised_sigma2) * d_mask) / torch.sum(d_mask)
#     loss2 = torch.sum((u + u_noised_sigma1 + u_noised_sigma2) * (1 - d_mask)) / torch.sum(1 - d_mask)

#     return loss1 + 0.1 * loss2

def gradient_loss(u, g, distance_map, batch_size, device,  threshold=0, sample_size=None, num_classes=None, onehot_label=None):
    # onehot_label [N,C,H,W]
    N = batch_size
    g = g.reshape(N, -1).to(device)
    u = u.reshape(N, -1).to(device)
    onehot_label = onehot_label.reshape(N, num_classes, -1).to(device)  # [N, C, HW]

    distance_map = distance_map.reshape(N, -1)
    distance_map = torch.as_tensor(distance_map, device=device)

    loss = torch.tensor(0.0, device=device)

    for b in range(N):
        g_batch = g[b,:]
        u_batch = u[b,:]
        distance_map_batch = distance_map[b,:]  # [HWD,]
        boundary = distance_map_batch <= threshold
        if not boundary.any():
            continue

        class_ids = torch.argmax(onehot_label[b], dim=0)  # [HW]
        class_ids = class_ids[boundary]
        g_valid = g_batch[boundary]
        u_valid = u_batch[boundary]
        
        for c in range(num_classes):
            idx = (class_ids == c).nonzero(as_tuple=True)[0]
            n = len(idx)
            if n < 2:
                continue
            
            max_pairs = n * (n - 1) // 2
            if sample_size is None:
                current_sample_size = max_pairs
            else:
                current_sample_size = min(sample_size, max_pairs)

            sampled_i = torch.randint(0, n, (current_sample_size,), device=device)
            sampled_j = torch.randint(0, n, (current_sample_size,), device=device)

            mask_valid = sampled_i != sampled_j
            sampled_i = sampled_i[mask_valid]
            sampled_j = sampled_j[mask_valid]
            if len(sampled_i) == 0:
                continue
            
            gi = g_valid[idx[sampled_i]]
            gj = g_valid[idx[sampled_j]]
            ui = u_valid[idx[sampled_i]]
            uj = u_valid[idx[sampled_j]]

            product = (gi - gj) * (ui - uj)
            # mask_positive = (product > 0).float()
            mask_positive = (product > 0).detach()
            positive_count = mask_positive.sum()
            if positive_count > 0:
                loss = loss + torch.sum(product * mask_positive) / positive_count
            
        del g_batch, u_batch, distance_map_batch
    loss = loss / N

    return loss
