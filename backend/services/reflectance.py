import numpy as np


def compute_reflectance_map(
    image: np.ndarray,
    roof_mask: np.ndarray,
):
    """
    Compute per-pixel reflectance for roof regions.

    Args:
        image: np.ndarray (H, W, 3), RGB, uint8 or float
        roof_mask: np.ndarray (H, W), uint8 or bool

    Returns:
        reflectance_map: np.ndarray (H, W), float32
    """
    image = image.astype(np.float32)

    if image.max() > 1.0:
        image = image / 255.0

    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    luminance = (
        0.2126 * R +
        0.7152 * G +
        0.0722 * B
    )

    reflectance_map = np.zeros_like(luminance, dtype=np.float32)
    reflectance_map[roof_mask > 0] = luminance[roof_mask > 0]

    return reflectance_map
