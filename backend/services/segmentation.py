import torch
import numpy as np

from backend.models.efficient_unet import get_efficientnet_unet
from backend.services.preprocessing import normalize_tile
from backend.services.tiling import tile_image, stitch_tiles


class RoofSegmentationService:
    def __init__(self, checkpoint_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Segmentation] Using device: {self.device}")

        self.model = get_efficientnet_unet()
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image: np.ndarray, threshold: float = 0.5):
        """
        Run tiled inference on full image.

        Args:
            image: np.ndarray (H, W, 3)
            threshold: sigmoid threshold

        Returns:
            binary_mask: np.ndarray (H, W), uint8
        """
        tile_preds = []
        tile_infos = []

        for tile, info in tile_image(image):
            tile_norm = normalize_tile(tile, method="imagenet")

            x = (
                torch.tensor(tile_norm)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .to(self.device)
            )

            logits = self.model(x)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

            tile_preds.append(probs)
            tile_infos.append(info)

        full_prob = stitch_tiles(
            tile_preds,
            tile_infos,
            full_shape=image.shape[:2],
        )

        binary_mask = (full_prob >= threshold).astype(np.uint8)
        return binary_mask
