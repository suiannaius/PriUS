import torch
import numpy as np
import json
import os
import re
from types import SimpleNamespace
from collections import Counter


def summarize_tensor(tensor, name="Tensor"):
    mean = tensor.mean().item()
    std = tensor.std().item()
    min_val = tensor.min().item()
    max_val = tensor.max().item()
    
    print(f"Summary of {name}:",
          f"Mean: {mean:.4f}",
          f"Std: {std:.4f}",
          f"Min: {min_val:.4f}",
          f"Max: {max_val:.4f}")
     
def save_args(args, filepath='config.json'):
    with open(filepath, 'w') as f:
        json.dump(vars(args), f, indent=4)

def load_args(filepath):
    with open(filepath, 'r') as f:
        config = json.load(f)
    return SimpleNamespace(**config)

def get_3d_index(m, H, W, D):
    a = m // (W * D)  # First dimension index
    b = (m % (W * D)) // D  # Second dimension index
    c = m % D  # Third dimension index
    return a, b, c

def get_2d_index(m, H, W):
    a = m // W  # First dimension index
    b = m % W  # Second dimension index
    return a, b

def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1 - (epoch) / max_epoch, power), 8)

def ensure_tensor(x, device):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    elif isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        raise TypeError("Unsupported input type")

def safe_item(x):
    return x.item() if isinstance(x, torch.Tensor) else x

def print_pairs_info(dataset):
    info_list = []
    modality_counter = Counter()
    center_counter = Counter()

    for img_path, lbl_path in dataset.pairs:
        img_name = os.path.basename(img_path).lower()

        modality = 'ct' if img_name.startswith('ct') else 'mr'
        modality_counter[modality] += 1

        match = re.search(r'(\d{4})', img_name)
        if match:
            num = match.group(1)
            first_digit = num[0]
            center_map = {'1':'A','2':'B','3':'C','5':'E','6':'F','7':'G'}
            center = center_map.get(first_digit, 'Unknown')
        else:
            center = 'Unknown'
        center_counter[center] += 1
        info_list.append((modality, center, os.path.basename(img_path), os.path.basename(lbl_path)))

    info_list = sorted(info_list, key=lambda x: (x[0], x[1], x[2]))

    print("===== Modality counts =====")
    for m, count in modality_counter.items():
        print(f"{m}: {count}")
    print("===== Center counts =====")
    for c, count in center_counter.items():
        print(f"{c}: {count}")


def check_training_statistics(use_prior=False):
    """Require statistics generated from training data before running a model."""
    required = [("zscore_stats_ct_centerA.json", "python get_data_stats.py")]
    if use_prior:
        required.append(("label_statistics.json", "python count_class_ratio.py"))
    missing = [(path, command) for path, command in required if not os.path.isfile(path)]
    if missing:
        instructions = "; ".join(f"{path}: run `{command}`" for path, command in missing)
        raise FileNotFoundError(
            "Missing training statistics. Set WHS_DATA_ROOT to your dataset directory "
            "and generate them from the training split: " + instructions +
            ". Run these commands from the repository root; reuse the same statistics for inference."
        )
