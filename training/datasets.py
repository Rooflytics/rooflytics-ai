import torch
from torch.utils.data import Dataset
import numpy as np

from backend.services.data_loader import load_geotiff
from backend.services.tiling import tile_image
from backend.services.preprocessing import normalize_tile


class RoofDataset(Dataset):
    def __init__(self, image_path, mask_path, max_tiles=50):
        image, _ = load_geotiff(image_path, is_mask=False)
        mask, _ = load_geotiff(mask_path, is_mask=True)

        # Sanity checks
        assert mask.ndim == 2, "Mask must be HW (single band)"
        assert mask.shape[:2] == image.shape[:2], "Image-mask size mismatch"

        self.tiles = []
        self.masks = []

        for tile, info in tile_image(image):
            y0, x0 = info["y_offset"], info["x_offset"]
            mask_tile = mask[y0:y0 + 512, x0:x0 + 512]

            m = mask_tile

            # Binary check
            assert m.max() <= 1.0 or m.max() == 255, "Mask not binary"

            if m.max() > 1:
                m = m / 255.0
            
            if m.sum() == 0:
              continue

            self.tiles.append(tile)
            self.masks.append(m)

            if len(self.tiles) >= max_tiles:
                break

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        x = normalize_tile(self.tiles[idx], method="per_image")
        y = self.masks[idx]

        x = torch.tensor(x).permute(2, 0, 1).float()
        y = torch.tensor(y).unsqueeze(0).float()

        return x, y
