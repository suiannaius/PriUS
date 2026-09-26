"""PriUS-only training loop; source objective and annealing schedule retained."""

import torch
import torch.nn.functional as F
from training.criterions import edl_loss
from training.principles import loss_gu, loss_nu, loss_du
from training.metrics import calculate_dice
from utilities.utils import (
    max_min_norm,
    adjust_learning_rate,
    sample_class_wise_noised_whole_images,
)


def train(model, dataloader, optimizer, args, current_epoch, device):
    model.train()
    totals = {"loss": 0.0, "edl": 0.0, "contrast": 0.0, "corruption_geometry": 0.0}
    dices = torch.zeros(args.num_classes - 1)
    if not len(dataloader):
        raise ValueError("Training set is smaller than batch_size with drop_last=True")
    active = current_epoch >= args.good_model_step
    upper = (
        min(
            int(
                args.sigma_upper_bound
                * (current_epoch - args.good_model_step)
                / (args.num_epochs - args.good_model_step)
                + 2
            ),
            args.sigma_upper_bound,
        )
        if active
        else None
    )
    annealing_start = torch.tensor(0.01, dtype=torch.float32)
    anneal = annealing_start * torch.exp(
        -torch.log(annealing_start)
        / (args.num_epochs - args.good_model_step)
        * (current_epoch - args.good_model_step)
    )
    for images, labels, _, distance, gradient in dataloader:
        adjust_learning_rate(optimizer, current_epoch, args.num_epochs, args.lr)
        images, labels, distance, gradient = [
            v.to(device) for v in (images, labels, distance, gradient)
        ]
        n, _, h, w = images.shape
        c = args.num_classes
        optimizer.zero_grad()
        evidence = (
            F.softplus(model(max_min_norm(images))).permute(0, 2, 3, 1).reshape(-1, c)
        )
        alpha = evidence + 1
        strength = alpha.sum(-1, keepdim=True)
        u = (c / strength).view(n, 1, h, w)
        prob = alpha / strength
        targets = labels.permute(0, 2, 3, 1).reshape(-1, c)
        base, *_ = edl_loss(
            targets,
            alpha,
            c,
            current_epoch,
            args.num_epochs,
            args.annealing_steps,
            device,
            args.loss_type,
            coef_cu=args.coef_cu,
        )
        gl = base.new_zeros(())
        nl = base.new_zeros(())
        if active:
            if args.gamma:
                sampled, x1, x2 = sample_class_wise_noised_whole_images(
                    images, device, upper_bound=upper, int_flag=True
                )
                us = [u]
                for x in (x1, x2):
                    en = F.softplus(
                        model(x).permute(0, 2, 3, 1).contiguous().view(-1, c)
                    )
                    us.append((c / (en + 1).sum(1, keepdim=True)).view(n, 1, h, w))
                if args.coef_sigma:
                    nl = nl + args.coef_sigma * loss_nu(
                        us, distance, [0.0, sampled["sigma1"], sampled["sigma2"]], args
                    )
                if args.coef_d:
                    # Preserve the external coefficient as well as the source loss's internal coefficient.
                    nl = (
                        nl
                        + args.coef_d
                        * sum(
                            loss_du(v, distance, labels, args, args.sample_size)
                            for v in us
                        )
                        / 3
                    )
            if args.beta:
                gl = loss_gu(u, gradient, distance, labels, args, args.sample_size)
        loss = base + args.beta * (gl * anneal) + args.gamma * (nl * anneal)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite loss; check images and principle support masks"
            )
        loss.backward()
        optimizer.step()
        for key, value in [
            ("loss", loss),
            ("edl", base),
            ("contrast", gl),
            ("corruption_geometry", nl),
        ]:
            totals[key] += value.item()
        dices += calculate_dice(targets.detach(), prob.detach(), 1e-5, device, c).cpu()
    return {
        **{k: v / len(dataloader) for k, v in totals.items()},
        "dice_per_class": (dices / len(dataloader)).tolist(),
    }
