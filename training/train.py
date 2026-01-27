import sys
from pathlib import Path

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
#sys.path.append(str(PROJECT_ROOT))

import torch
import yaml
from torch.utils.data import DataLoader

from backend.models.unet_scratch import UNet
from training.datasets import RoofDataset
from training.losses import BCEDiceLoss
from training.engine import train_one_epoch
from torch.utils.data import ConcatDataset

from pathlib import Path

DATA_DIR = Path("data/small")
IMAGE_DIR = DATA_DIR / "image"
MASK_DIR = DATA_DIR / "label"

image_paths = sorted(IMAGE_DIR.glob("*.tif"))
mask_paths = sorted(MASK_DIR.glob("*.tif"))

assert len(image_paths) == len(mask_paths) > 0

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    with open("training/configs/scratch.yaml") as f:
        cfg = yaml.safe_load(f)

    datasets = [
    RoofDataset(img, msk, max_tiles=200)
    for img, msk in zip(image_paths[:10], mask_paths[:10])
    ]

    dataset = ConcatDataset(datasets)

    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
    )

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = BCEDiceLoss()

    for epoch in range(cfg["epochs"]):
        loss = train_one_epoch(model, loader, optimizer, loss_fn, device)
        print(f"Epoch {epoch+1}/{cfg['epochs']} — Loss: {loss:.4f}")

    torch.save(model.state_dict(), "scratch_unet.pth")
    print("Model saved as scratch_unet.pth")


if __name__ == "__main__":
    main()
