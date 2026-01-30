import rasterio
from rasterio.enums import Resampling


def load_geotiff(path: str, is_mask: bool = False):
    """
    Load GeoTIFF.

    Args:
        path: path to tif
        is_mask: if True, loads single-band mask

    Returns:
        image: np.ndarray
        meta: rasterio metadata
    """
    with rasterio.open(path) as src:
        if is_mask:
            image = src.read(
                1,
                out_dtype="float32",
                resampling=Resampling.nearest,
            )
        else:
            if src.count < 3:
                raise ValueError("RGB GeoTIFF must have at least 3 bands")

            image = src.read(
                indexes=[1, 2, 3],
                out_dtype="float32",
                resampling=Resampling.nearest,
            )

        meta = src.meta.copy()

    if not is_mask:
        image = image.transpose(1, 2, 0)  # (H, W, 3)

    return image, meta
