import numpy as np
from typing import Iterator, Tuple


def tile_image(
    image: np.ndarray,
    tile_size: int = 512,
    overlap: int = 0,
) -> Iterator[Tuple[np.ndarray, dict]]:
    """
    Tile image into patches while preserving spatial indices.

    Returns:
        tile: (tile_size, tile_size, C)
        info: dict with spatial metadata
    """
    h, w, c = image.shape
    stride = tile_size - overlap

    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            tile = image[y:y + tile_size, x:x + tile_size, :]

            info = {
                "x_offset": x,
                "y_offset": y,
                "height": tile_size,
                "width": tile_size,
            }

            yield tile, info

def stitch_tiles(
    tile_preds,
    tile_infos,
    full_shape,
):
    """
    Reconstruct full-size mask from tile predictions.

    Args:
        tile_preds: list of np.ndarray (H, W) or (H, W, 1)
        tile_infos: list of dicts with x_offset, y_offset
        full_shape: (H, W)

    Returns:
        full_mask: np.ndarray (H, W)
    """
    import numpy as np

    full_mask = np.zeros(full_shape, dtype=np.float32)

    for pred, info in zip(tile_preds, tile_infos):
        y = info["y_offset"]
        x = info["x_offset"]

        if pred.ndim == 3:
            pred = pred.squeeze(-1)

        h, w = pred.shape
        full_mask[y:y + h, x:x + w] = pred

    return full_mask