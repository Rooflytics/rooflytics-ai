import random
import torch
from torch.utils.data import Dataset
import numpy as np

from backend.services.data_loader import load_geotiff
from backend.services.tiling import tile_image
from backend.services.preprocessing import normalize_tile


class RoofDatasetProduction(Dataset):
    """
    Production dataset with:
    - ImageNet normalization
    - Random tile sampling
    - Foreground + background balance
    """

    def __init__(
        self,
        image_path,
        mask_path,
        max_tiles=300,
        fg_ratio=0.7,
        seed=42,
    ):
        random.seed(seed)

        image, _ = load_geotiff(image_path, is_mask=False)
        mask, _ = load_geotiff(mask_path, is_mask=True)

        assert mask.shape == image.shape[:2]

        fg_tiles = []
        bg_tiles = []

        for tile, info in tile_image(image):
            y0, x0 = info["y_offset"], info["x_offset"]
            mask_tile = mask[y0:y0 + 512, x0:x0 + 512]

            if mask_tile.sum() > 0:
                fg_tiles.append((tile, mask_tile))
            else:
                bg_tiles.append((tile, mask_tile))

        # ---- Balanced sampling ----
        num_fg = int(max_tiles * fg_ratio)
        num_bg = max_tiles - num_fg

        fg_samples = random.sample(
            fg_tiles,
            min(len(fg_tiles), num_fg),
        )
        bg_samples = random.sample(
            bg_tiles,
            min(len(bg_tiles), num_bg),
        )

        samples = fg_samples + bg_samples
        random.shuffle(samples)

        self.tiles, self.masks = zip(*samples)

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        x = normalize_tile(self.tiles[idx], method="imagenet")
        y = self.masks[idx]

        x = torch.tensor(x).permute(2, 0, 1).float()
        y = torch.tensor(y).unsqueeze(0).float()

        return x, y
