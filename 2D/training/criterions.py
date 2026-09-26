"""Evidential segmentation objective from the source implementation."""

import torch


def Dice(y_true, y_pred, epsilon, device):
    """
    Input tensor:
    y_true: [NHWD, C]
    y_pred: [NHWD, C]
    alpha: [NHWD, C]
    """
    y_true = y_true.to(device)  # [NHWD, C]
    y_pred = y_pred.to(device)  # [NHWD, C]
    smooth = (
        torch.zeros(y_pred.size(-1), dtype=torch.float32).fill_(0.00001).to(device)
    )  # [C, ]
    ones = torch.ones(y_pred.shape).to(device)  # [NHWD, C]
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
    dice = num / (den + smooth)  # (C, )

    return dice


def DiceLoss(y_true, y_pred, epsilon, device):
    return 1 - Dice(y_true, y_pred, epsilon, device)


def KL(alp, c, device, beta=None):
    if beta is None:  # uniform prior
        beta = torch.ones((1, c), device=device)
        assert torch.all(alp >= 1), "alp needs to be greater than or equal to 1."
    else:
        beta = beta.to(device).view(1, c)
    alp = alp.to(device)  # [NHWD,C]
    S_alp = torch.sum(alp, dim=1, keepdim=True)  # [NHWD,1]

    S_beta = torch.sum(beta, dim=1, keepdim=True)  # [1,1]
    lnB = torch.lgamma(S_alp) - torch.sum(torch.lgamma(alp), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alp)
    dg1 = torch.digamma(alp)
    kl = torch.sum((alp - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni

    return kl


def edl_loss(
    y_true,
    alpha,
    num_classes,
    current_epoch,
    total_epoch,
    annealing_steps,
    device,
    loss_type,
    coef_cu=0.0,
    prior=None,
    **kwargs,
):
    assert loss_type in ["log", "digamma", "mse"], (
        f"'loss_type' should be 'log', 'digamma' or 'mse', but got {loss_type}."
    )
    y_true = y_true.to(device)  # [NHWD, 4]
    alpha = alpha.to(device)  # [NHWD, 4]
    S = torch.sum(alpha, dim=-1, keepdim=True)  # [NHWD, 1]
    prob = alpha / S  # [NHWD, 4]
    if loss_type == "log":
        L_ACE = torch.sum(
            torch.mul(y_true, (torch.log(S) - torch.log(alpha))), dim=-1, keepdim=True
        )  # [NHWD, 1]
    elif loss_type == "digamma":
        L_ACE = torch.sum(
            torch.mul(y_true, (torch.digamma(S) - torch.digamma(alpha))),
            dim=-1,
            keepdim=True,
        )  # [NHWD, 1]
    elif loss_type == "mse":
        L_err = torch.sum((y_true - prob) ** 2, dim=-1, keepdim=True)  # [NHWD, 1]
        L_var = alpha * (S - alpha) / (S * S * (S + 1))  # [NHWD, 4]
        L_ACE = torch.sum(L_err + L_var, dim=-1, keepdim=True)  # [NHWD, 1]
    alp = (alpha - 1) * (1 - y_true) + 1  # [NHWD, 4]
    annealing_coef = min(1, current_epoch / annealing_steps)
    L_KL = KL(alp, num_classes, device, beta=prior)  # [NHWD, 1]
    L_DICE = DiceLoss(y_true, prob, epsilon=1e-3, device=device)  # [C,]
    annealing_start = torch.tensor(0.01, dtype=torch.float32)
    annealing_AU = annealing_start * torch.exp(
        -torch.log(annealing_start) / total_epoch * current_epoch
    )  # from 'annealing_start' to '1'
    # AU Loss
    pred_scores, pred_cls = torch.max(alpha / S, 1, keepdim=True)
    uncertainty = num_classes / S
    target = torch.argmax(y_true, dim=1, keepdims=True)
    acc_match = torch.reshape(torch.eq(pred_cls, target).float(), (-1, 1))  # [NHWD, 1]
    acc_uncertain = -pred_scores * torch.log(1 - uncertainty + 1e-5)  # [NHWD, 1]
    inacc_certain = -(1 - pred_scores) * torch.log(uncertainty + 1e-5)
    L_AU = (
        annealing_AU * acc_match * acc_uncertain
        + (1 - annealing_AU) * (1 - acc_match) * inacc_certain
    )
    loss = torch.mean(
        L_ACE
        + kwargs.get("coef_KL", 1.0) * annealing_coef * L_KL
        + (1 - annealing_AU) * L_DICE
        + coef_cu * L_AU
    )
    L_ACE = torch.mean(L_ACE)
    L_KL = torch.mean(L_KL)
    L_DICE = torch.mean(L_DICE)
    L_AU = torch.mean(L_AU)

    return loss, L_ACE, L_KL, L_DICE, L_AU
