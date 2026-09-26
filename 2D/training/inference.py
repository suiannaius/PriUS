"""Complete-batch evaluation with per-image records and optional UCC/UR."""

import numpy as np
import torch
import torch.nn.functional as F
from training.dataset import sample_names
from training.metrics import calculate_dice, soft_hd95
from utilities.utils import max_min_norm, generate_noisy_images
from utilities.count_pixels import (
    count_pixels_grad,
    count_pixels_sigma,
    count_pixels_d_chunk,
)


def inference(model, dataloader, args, device):
    model.eval()
    records = []
    names = sample_names(dataloader.dataset)
    offset = 0
    with torch.inference_mode():
        for batch_idx, (images, labels, spacing, distance, gradient) in enumerate(
            dataloader
        ):
            images, labels, distance, gradient = [
                v.to(device) for v in (images, labels, distance, gradient)
            ]
            n, _, h, w = images.shape
            c = args.num_classes
            alpha = F.softplus(model(max_min_norm(images))) + 1
            prob = alpha / alpha.sum(1, keepdim=True)
            u = c / alpha.sum(1, keepdim=True)
            pred = prob.argmax(1)
            truth = labels.argmax(1)
            if args.uncertainty_metrics:
                x1, _, s1, _ = generate_noisy_images(
                    images, device, seed=1, sigma_set=10
                )
                x2, _, s2, _ = generate_noisy_images(
                    images, device, seed=1, sigma_set=args.sigma_upper_bound
                )
                us = [u] + [
                    c / (F.softplus(model(x)) + 1).sum(1, keepdim=True)
                    for x in (x1, x2)
                ]
            for j in range(n):
                targets = labels[j].permute(1, 2, 0).reshape(-1, c)
                pj = prob[j].permute(1, 2, 0).reshape(-1, c)
                dice = calculate_dice(targets, pj, 1e-5, device, c).tolist()
                # SimpleITK spacing is (x,y); arrays are (y,x).
                sp = (
                    spacing[j : j + 1].numpy()[:, ::-1].copy()
                    if args.dataset == "ACDC"
                    else None
                )
                hd = soft_hd95(
                    truth[j : j + 1], pred[j : j + 1], sp, num_classes=c
                ).tolist()
                r = dict(
                    sample=names[offset + j],
                    dice_per_class=dice,
                    hd95_per_class=hd,
                    dice=float(np.mean(dice)),
                    hd95=float(np.mean(hd)),
                )
                if args.uncertainty_metrics:
                    dj = distance[j : j + 1]
                    lj = labels[j : j + 1]
                    uj = [v[j : j + 1].reshape(-1, 1) for v in us]
                    rg, cg = count_pixels_grad(
                        uj[0], gradient[j : j + 1], dj, lj, 1, device, threshold=0
                    )
                    rs, cs = count_pixels_sigma(
                        uj[1].reshape(-1),
                        uj[2].reshape(-1),
                        s1,
                        s2,
                        dj.reshape(-1),
                        device,
                        threshold=args.threshold_sigma,
                    )
                    gd = [
                        count_pixels_d_chunk(
                            v,
                            dj,
                            lj,
                            1,
                            device,
                            threshold=args.threshold_d,
                            threshold_min_dif=3,
                        )
                        for v in uj
                    ]
                    r.update(
                        ucc_grad=float(cg),
                        ur_grad=float(rg),
                        ucc_sigma=float(cs),
                        ur_sigma=float(rs),
                        ucc_d=float(np.mean([v[1] for v in gd])),
                        ur_d=float(np.mean([v[0] for v in gd])),
                    )
                if args.save_predictions:
                    dest = args.output_dir / f"predictions/fold_{args.fold}"
                    dest.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        dest / (r["sample"] + ".npz"),
                        prediction=pred[j].cpu().numpy().astype(np.uint8),
                        uncertainty=u[j, 0].cpu().numpy(),
                    )
                records.append(r)
            offset += n
            print(f"Evaluated {offset}/{len(names)} samples", flush=True)
    if offset != len(names):
        raise RuntimeError("Evaluation did not cover every sample")
    keys = ["dice", "hd95"] + (
        ["ucc_grad", "ucc_sigma", "ucc_d", "ur_grad", "ur_sigma", "ur_d"]
        if args.uncertainty_metrics
        else []
    )
    values = np.array([[r[k] for k in keys] for r in records])
    if not np.isfinite(values).all():
        raise FloatingPointError("Evaluation produced non-finite metrics")
    summary = {k: float(values[:, i].mean()) for i, k in enumerate(keys)}
    return dict(
        dataset=args.dataset,
        fold=args.fold,
        samples_evaluated=offset,
        aggregation="per-image/slice foreground macro segmentation metrics, then sample mean",
        hd95_unit="mm (2D slice surfaces)"
        if args.dataset == "ACDC"
        else "pixels at 256x256",
        summary=summary,
        per_sample=records,
    )
