import numpy as np
import cv2


def clean_roof_mask(
    mask: np.ndarray,
    min_area: int = 100,
    kernel_size: int = 3,
):
    """
    Clean raw roof segmentation mask.

    Args:
        mask: np.ndarray (H, W), uint8 (0 or 1)
        min_area: minimum connected component area to keep
        kernel_size: morphological kernel size

    Returns:
        cleaned_mask: np.ndarray (H, W), uint8
    """
    # Ensure uint8
    mask = mask.astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    # 1️⃣ Close small holes inside roofs
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 2️⃣ Remove isolated noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 3️⃣ Remove tiny connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    cleaned = np.zeros_like(mask)

    for i in range(1, num_labels):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 1

    return cleaned
