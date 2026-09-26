"""Evaluate one ACDC fold or a supplied ISIC image/mask split."""

import torch
from torch.utils.data import DataLoader
from model.models import Unet2D
from training.dataset import make_dataset
from training.inference import inference
from utilities.config import parse_args, seed_everything, save_json


def main():
    a = parse_args()
    seed_everything(a.seed)
    model = Unet2D(a.num_modalities, a.num_classes).to(a.device)
    model.load_state_dict(
        torch.load(a.checkpoint, map_location=a.device, weights_only=True)
    )
    ds = make_dataset(a, a.fold)
    loader = DataLoader(
        ds,
        batch_size=a.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=a.num_workers,
        pin_memory=a.device.startswith("cuda"),
    )
    result = inference(model, loader, a, a.device)
    result["checkpoint"] = str(a.checkpoint)
    save_json(a.output_dir / f"results/fold_{a.fold}.json", result)
    save_json(a.output_dir / f"results/fold_{a.fold}_config.json", vars(a))
    print(result["summary"])


if __name__ == "__main__":
    main()
