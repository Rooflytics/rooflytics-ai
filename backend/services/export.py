import rasterio
from rasterio.enums import Compression
import numpy as np


def export_mask_geotiff(
    mask: np.ndarray,
    reference_meta: dict,
    output_path: str,
):
    """
    Export binary mask as GeoTIFF using reference metadata.

    Args:
        mask: np.ndarray (H, W), uint8 or bool
        reference_meta: metadata from input GeoTIFF
        output_path: path to save output
    """
    meta = reference_meta.copy()

    meta.update({
        "count": 1,
        "dtype": rasterio.uint8,
        "compress": Compression.lzw,
    })

    mask = mask.astype("uint8")

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(mask, 1)