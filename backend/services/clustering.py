import numpy as np
import cv2
from sklearn.cluster import KMeans


def extract_roof_reflectance(
    roof_mask: np.ndarray,
    reflectance_map: np.ndarray,
    min_pixels: int = 50,
):
    """
    Extract per-roof reflectance statistics.

    Args:
        roof_mask: np.ndarray (H, W), uint8
        reflectance_map: np.ndarray (H, W), float
        min_pixels: minimum pixels to consider a roof valid

    Returns:
        roof_stats: list of dicts
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        roof_mask.astype(np.uint8),
        connectivity=8,
    )

    roof_stats = []

    for i in range(1, num_labels):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_pixels:
            continue

        pixels = reflectance_map[labels == i]
        pixels = pixels[pixels > 0]

        if len(pixels) == 0:
            continue

        roof_stats.append({
            "label": i,
            "area_pixels": area,
            "mean_reflectance": float(pixels.mean()),
            "median_reflectance": float(np.median(pixels)),
        })

    return roof_stats


def cluster_roofs_by_reflectance(
    roof_stats,
):
    """
    Cluster roofs into hot/cool based on reflectance.

    Returns:
        roof_stats with cluster labels added
    """
    values = np.array(
        [[r["mean_reflectance"]] for r in roof_stats],
        dtype=np.float32,
    )

    kmeans = KMeans(
        n_clusters=2,
        random_state=42,
        n_init=10,
    )
    clusters = kmeans.fit_predict(values)

    # Determine which cluster is "cool"
    cluster_means = {
        c: values[clusters == c].mean()
        for c in [0, 1]
    }

    cool_cluster = max(cluster_means, key=cluster_means.get)

    for r, c in zip(roof_stats, clusters):
        r["cluster"] = int(c)
        r["type"] = "cool" if c == cool_cluster else "hot"

    return roof_stats


def create_thermal_cluster_mask(
    roof_mask: np.ndarray,
    roof_stats,
):
    """
    Create raster mask with thermal clusters.

    Values:
        0 = background
        1 = hot roof
        2 = cool roof
    """
    thermal_mask = np.zeros_like(roof_mask, dtype=np.uint8)

    # Build lookup: label -> type
    label_to_type = {
        r["label"]: r["type"]
        for r in roof_stats
    }

    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(
        roof_mask.astype(np.uint8),
        connectivity=8,
    )

    for lbl in range(1, num_labels):
        if lbl not in label_to_type:
            continue

        if label_to_type[lbl] == "hot":
            thermal_mask[labels == lbl] = 1
        else:
            thermal_mask[labels == lbl] = 2

    return thermal_mask