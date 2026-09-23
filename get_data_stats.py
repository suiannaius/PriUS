import os
import json
import torch
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from training.dataset import WHS_Dataset


def compute_percentile_zscore_stats_from_dataset(dataset: WHS_Dataset,):
    """
    Compute dataset-level percentile-based z-score statistics
    using per-volume percentiles to avoid excessive memory usage.
    """
    assert dataset.mode == 'train'
    assert dataset.normalize is False, "'normalize' must be False when computing data stats."

    low_list = []
    high_list = []
    mean_list = []
    std_list = []
    voxel_count = 0

    print(f"\n[Stats] modality = {dataset.modality_filter}")
    print(f"[Stats] center   = {dataset.center_filter}")
    print(f"[Stats] #cases   = {len(dataset.pairs)}")

    for idx, (img_path, _) in enumerate(tqdm(dataset.pairs, desc="Computing per-volume stats"), start=1):
        img_itk = sitk.ReadImage(img_path)
        img_rs = dataset._resample_to_spacing(img_itk, dataset.new_spacing, is_label=False)

        img = sitk.GetArrayFromImage(img_rs).astype(np.float32)
        voxels = torch.from_numpy(img.reshape(-1))

        # per-volume percentiles
        low_i  = torch.quantile(voxels, 0.005)
        high_i = torch.quantile(voxels, 0.995)

        clipped = torch.clamp(voxels, low_i, high_i)

        mean_i = clipped.mean()
        std_i  = clipped.std(unbiased=False)
        
        case_name = os.path.basename(img_path)
        print(
            f"[{idx:03d}] {case_name} | "
            f"low={low_i:.4f}, high={high_i:.4f}, "
            f"mean={mean_i:.4f}, std={std_i:.4f}"
        )

        low_list.append(low_i)
        high_list.append(high_i)
        mean_list.append(mean_i)
        std_list.append(std_i)
        voxel_count += voxels.numel()

    # dataset-level aggregation
    low  = torch.mean(torch.stack(low_list))
    high = torch.mean(torch.stack(high_list))
    mean = torch.mean(torch.stack(mean_list))
    std  = torch.mean(torch.stack(std_list))

    stats = {
        "low":  float(low),
        "high": float(high),
        "mean": float(mean),
        "std":  float(std),
        "num_voxels": int(voxel_count),
        "modality": dataset.modality_filter,
        "center": dataset.center_filter,
        "spacing": dataset.new_spacing,
        "target_size": dataset.target_size,
        "patch_size": dataset.patch_size,
    }

    return stats


def main():
    dataset = WHS_Dataset(
        root_dir=os.environ["WHS_DATA_ROOT"],
        num_classes=8,
        mode='train',
        patch_size=(64, 64, 64),
        modality_filter='ct',
        center_filter='A',
        target_size=(150, 150, 150),
        new_spacing=(2.0, 2.0, 2.0),
        normalize=False
    )

    stats = compute_percentile_zscore_stats_from_dataset(dataset)

    out_path = (
        f"zscore_stats_"
        f"{stats['modality']}_"
        f"center{stats['center']}.json"
    )

    with open(out_path, "w") as f:
        json.dump(stats, f, indent=4)

    print("\n[Done] Percentile z-score statistics saved to:")
    print(out_path)
    print(json.dumps(stats, indent=4))


if __name__ == "__main__":
    main()
