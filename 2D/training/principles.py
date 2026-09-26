"""Contrast, corruption, and geometry supervision from US_ELEGANT."""

import torch
import torch.nn.functional as F
from utilities.geometry import aggregate_along_normal, sigmoid_indicator, penalty


def loss_gu(u, g, d, labels, args, sample_size):
    """
    u,g: [N,1,D,H,W]
    d:   [N,D,H,W]
    labels: [N,C,D,H,W]
    """
    device = u.device
    d = d.to(device)
    g = g.to(device)
    labels = labels.to(device)

    u_tilde, g_tilde = aggregate_along_normal(u, g, kernel_size=args.normal_kernel)

    # flatten
    u_tilde = u_tilde.view(-1)
    g_tilde = g_tilde.view(-1)
    d = d.view(-1)
    lbl = torch.argmax(labels, dim=1).view(-1)

    idx = torch.randperm(u_tilde.numel(), device=device)[:sample_size]
    i, j = idx[::2], idx[1::2]

    same_class = lbl[i] == lbl[j]

    w = sigmoid_indicator(args.d_g - torch.max(d[i], d[j]), alpha=args.alpha * 10.0)

    violation = (u_tilde[i] - u_tilde[j]) * (g_tilde[i] - g_tilde[j])
    loss = w * penalty(F.relu(violation)) * same_class.float()

    loss = loss.sum() / (w * same_class).sum().clamp_min(1e-12)

    return loss


def loss_nu(u_list, d, sigmas, args):
    """
    u_list: list of [N,1,D,H,W], n = 0,1,2
    d:      [N,D,H,W]
    """
    device = u_list[0].device
    d = d.to(device)
    d = d.view(-1)
    k = args.patch_kernel
    pad = k // 2

    u_bar_list = [
        F.avg_pool2d(u, kernel_size=k, stride=1, padding=pad) for u in u_list
    ]  # 每个都是 [N,1,D,H,W]
    loss = 0.0
    for n in [1, 2]:
        du = (u_bar_list[n] - u_bar_list[n - 1].clone().detach()).view(-1)
        w = sigmoid_indicator(args.d_n - d, alpha=args.alpha)
        violation = -(sigmas[n] - sigmas[n - 1]) * du
        loss += w * penalty(F.relu(violation)) / w.sum().clamp_min(1e-12)
    return loss.sum()


def loss_du(u, d, labels, args, sample_size):
    device = u.device
    d = d.to(device)
    labels = labels.to(device)

    u = u.view(-1)
    d = d.view(-1)

    lbl = torch.argmax(labels, dim=1).view(-1)

    idx = torch.randperm(u.numel(), device=device)[:sample_size]
    i, j = idx[::2], idx[1::2]

    same_class = (lbl[i] == lbl[j]).float()

    tij = sigmoid_indicator(d[i] - args.d_f, alpha=args.alpha) * sigmoid_indicator(
        d[j] - args.d_f, alpha=args.alpha
    )

    wij = sigmoid_indicator(torch.abs(d[i] - d[j]) - args.d_eps, alpha=args.alpha)

    sij = args.coef_d * wij * (1 - tij) * (d[i] - d[j]) + tij * args.coef_far

    u_j_choose = (1 - 2 * tij) * u[j]

    violation = sij * (u[i] - u_j_choose)
    loss_raw = penalty(F.relu(violation)) * same_class

    w_near = wij * same_class * (1 - tij)
    w_far = same_class * tij
    normalizer = w_near.sum() + w_far.sum()

    loss = loss_raw.sum() / normalizer.clamp_min(1e-12)
    return loss
