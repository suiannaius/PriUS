"""Shared command-line configuration for the 2D entry points."""

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch


def parse_args(training=False):
    p = argparse.ArgumentParser(description="PriUS 2D: ACDC / ISIC")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--data-dir",
        type=Path,
        help="ACDC directory containing patient001 ... patient100",
    )
    p.add_argument("--image-dir", type=Path, help="ISIC image directory")
    p.add_argument("--mask-dir", type=Path, help="ISIC segmentation directory")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--fold", type=int, choices=range(5), default=None if training else 0
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--seed", type=int)
    if training:
        p.add_argument("--num-epochs", type=int)
        p.add_argument("--good-model-step", type=int)
        p.add_argument("--sample-size", type=int)
    else:
        p.add_argument("--checkpoint", type=Path, required=True)
        p.add_argument(
            "--uncertainty-metrics",
            action="store_true",
            help="Also evaluate UCC/UR; requires two noisy forwards",
        )
        p.add_argument(
            "--save-predictions",
            action="store_true",
            help="Save per-slice/image NPZ prediction and uncertainty",
        )
    cli = p.parse_args()
    cfg = json.loads(cli.config.read_text())
    cfg.update({k: v for k, v in vars(cli).items() if v is not None})
    cfg.setdefault("fold", None)
    a = SimpleNamespace(**cfg)
    if a.dataset not in ("ACDC", "ISIC"):
        p.error("dataset must be ACDC or ISIC")
    if a.dataset == "ACDC" and not getattr(a, "data_dir", None):
        p.error("ACDC requires --data-dir")
    if a.dataset == "ISIC" and (
        not getattr(a, "image_dir", None) or not getattr(a, "mask_dir", None)
    ):
        p.error("ISIC requires --image-dir and --mask-dir")
    if a.dataset == "ISIC" and a.fold not in (None, 0):
        p.error("ISIC supports only fold 0")
    if (a.num_modalities, a.num_classes) != ((1, 4) if a.dataset == "ACDC" else (3, 2)):
        p.error("configuration channels/classes do not match dataset")
    if a.batch_size < 1 or a.threads < 1 or a.num_workers < 0:
        p.error("invalid batch size, threads, or worker count")
    if training and not 0 <= a.good_model_step < a.num_epochs:
        p.error("require 0 <= good_model_step < num_epochs")
    if training and (a.sample_size < 2 or a.sample_size % 2):
        p.error("sample_size must be positive and even")
    if a.annealing_steps <= 0:
        p.error("annealing_steps must be positive")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.threads)
    return a


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")
