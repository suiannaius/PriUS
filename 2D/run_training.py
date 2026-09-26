"""Train PriUS on ACDC (five folds) or ISIC (one supplied training split)."""

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.models import Unet2D
from training.dataset import make_dataset, sample_names
from training.train import train
from utilities.config import parse_args, seed_everything, save_json


def main():
    a = parse_args(training=True)
    folds = (
        ([a.fold] if a.fold is not None else list(range(5)))
        if a.dataset == "ACDC"
        else [0]
    )
    if any((a.output_dir / f"saved_models/fold_{f}.pth").exists() for f in folds):
        raise FileExistsError(
            "Checkpoints already exist; select a new output directory"
        )
    save_json(a.output_dir / "config.json", vars(a))
    for fold in folds:
        seed_everything(a.seed + fold)
        ds = make_dataset(a, fold, training=True)
        loader = DataLoader(
            ds,
            batch_size=a.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=a.num_workers,
            pin_memory=a.device.startswith("cuda"),
        )
        model = Unet2D(a.num_modalities, a.num_classes).to(a.device)
        # The original initializer affects 3D layers only, so 2D uses PyTorch defaults.
        optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
        checkpoint = a.output_dir / f"saved_models/fold_{fold}.pth"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        save_json(a.output_dir / f"logging/fold_{fold}_samples.json", sample_names(ds))
        history = []
        with SummaryWriter(str(a.output_dir / f"runs/fold_{fold}")) as writer:
            for epoch in range(a.num_epochs):
                metrics = train(model, loader, optimizer, a, epoch, a.device)
                history.append(dict(epoch=epoch + 1, **metrics))
                for key in ["loss", "edl", "contrast", "corruption_geometry"]:
                    writer.add_scalar(key, metrics[key], epoch + 1)
                print(
                    f"fold={fold} epoch={epoch + 1}/{a.num_epochs} loss={metrics['loss']:.6f}",
                    flush=True,
                )
                if epoch == a.num_epochs - 1 or epoch % 10 == 0:
                    torch.save(model.state_dict(), checkpoint)
                save_json(a.output_dir / f"logging/fold_{fold}.json", history)


if __name__ == "__main__":
    main()
