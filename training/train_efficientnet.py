import torch
import yaml
from torch.utils.data import DataLoader, ConcatDataset
from torch.amp import autocast, GradScaler
from pathlib import Path

from backend.models.efficient_unet import get_efficientnet_unet
from training.dataset_production import RoofDatasetProduction
from training.losses import BCEDiceLoss


def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        with autocast("cuda"):
            preds = model(x)
            loss = loss_fn(preds, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    with open("training/configs/efficientnet.yaml") as f:
        cfg = yaml.safe_load(f)

    DATA_DIR = Path(cfg["data_dir"])
    IMAGE_DIR = DATA_DIR / "image"
    MASK_DIR = DATA_DIR / "label"

    image_paths = sorted(IMAGE_DIR.glob("*.tif"))
    mask_paths = sorted(MASK_DIR.glob("*.tif"))

    datasets = [
        RoofDatasetProduction(
            img,
            msk,
            max_tiles=cfg["max_tiles"],
            fg_ratio=cfg["fg_ratio"],
            seed=cfg["seed"],
        )
        for img, msk in zip(image_paths[:cfg["num_images"]],
                            mask_paths[:cfg["num_images"]])
    ]

    dataset = ConcatDataset(datasets)

    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    model = get_efficientnet_unet().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=1e-4,
    )

    loss_fn = BCEDiceLoss()
    scaler = GradScaler("cuda")

    for epoch in range(cfg["epochs"]):
        loss = train_one_epoch(
            model, loader, optimizer, loss_fn, scaler, device
        )
        print(f"Epoch {epoch+1}/{cfg['epochs']} — Loss: {loss:.4f}")

    torch.save(model.state_dict(), "efficientnet_unet.pth")
    print("Model saved as efficientnet_unet.pth")


if __name__ == "__main__":
    main()
