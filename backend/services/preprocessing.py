import numpy as np
import cv2


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_tile(
    tile: np.ndarray,
    method: str = "imagenet",
):
    """
    Normalize a single tile.

    Args:
        tile: np.ndarray (H, W, 3), values 0–255 or 0–1
        method: "imagenet" or "per_image"

    Returns:
        normalized tile (float32)
    """
    tile = tile.astype(np.float32)

    # Scale to 0–1 if needed
    if tile.max() > 1.0:
        tile = tile / 255.0

    if method == "imagenet":
        tile = (tile - IMAGENET_MEAN) / IMAGENET_STD

    elif method == "per_image":
        mean = tile.mean(axis=(0, 1), keepdims=True)
        std = tile.std(axis=(0, 1), keepdims=True) + 1e-6
        tile = (tile - mean) / std

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return tile


def detect_shadows(tile: np.ndarray, threshold: float = 0.15):
    """
    Detect shadow regions using luminance thresholding.

    Args:
        tile: np.ndarray (H, W, 3) in 0–1 range
        threshold: luminance threshold

    Returns:
        shadow_mask: np.ndarray (H, W), bool
    """
    if tile.max() > 1.0:
        tile = tile / 255.0

    # Convert to grayscale luminance
    gray = (
        0.2126 * tile[:, :, 0]
        + 0.7152 * tile[:, :, 1]
        + 0.0722 * tile[:, :, 2]
    )

    shadow_mask = gray < threshold
    return shadow_mask


def morphological_cleanup(
    mask: np.ndarray,
    kernel_size: int = 3,
):
    """
    Clean binary mask using morphological operations.

    Args:
        mask: np.ndarray (H, W), bool or 0/1
        kernel_size: size of morphological kernel

    Returns:
        cleaned_mask: np.ndarray (H, W), bool
    """
    import cv2
    mask = mask.astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    # Close small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Remove isolated noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask.astype(bool)


def preprocess_tile(
    tile: np.ndarray,
    norm_method: str = "imagenet",
):
    """
    Full preprocessing pipeline for a single tile.

    Returns:
        processed_tile: np.ndarray (H, W, 3)
        shadow_mask: np.ndarray (H, W), bool
    """
    tile_norm = normalize_tile(tile, method=norm_method)
    shadow_mask = detect_shadows(tile)
    shadow_mask = morphological_cleanup(shadow_mask)

    return tile_norm, shadow_mask
