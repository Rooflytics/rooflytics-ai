import numpy as np
import cv2


def compute_roof_areas(
    roof_mask: np.ndarray,
    pixel_area_m2: float,
):
    """
    Compute roof areas in square meters.

    Args:
        roof_mask: np.ndarray (H, W), uint8
        pixel_area_m2: area of one pixel in m²

    Returns:
        dict: label -> area_m2
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        roof_mask.astype(np.uint8),
        connectivity=8,
    )

    roof_areas = {}

    for i in range(1, num_labels):
        area_pixels = stats[i, cv2.CC_STAT_AREA]
        roof_areas[i] = area_pixels * pixel_area_m2

    return roof_areas


def estimate_cooling_savings(
    roof_stats,
    roof_areas,
    constants,
):
    """
    Estimate annual cooling energy & CO2 savings.

    Args:
        roof_stats: list of dicts (from clustering)
        roof_areas: dict label -> area_m2
        constants: energy model constants

    Returns:
        list of per-roof energy results
    """
    results = []

    for r in roof_stats:
        label = r["label"]
        area = roof_areas.get(label, 0)

        if area == 0:
            continue

        # Hot roofs benefit more from retrofitting
        if r["type"] == "hot":
            delta_reflectance = 0.4  # assumed improvement
        else:
            delta_reflectance = 0.1

        USAGE_FACTOR = constants.get("USAGE_FACTOR", 0.025)
        MAX_KWH_PER_ROOF = constants.get("MAX_KWH_PER_ROOF", 5000)

        annual_energy_kwh = (
            area
            * constants["SOLAR_IRRADIANCE"]
            * constants["SUNLIGHT_HOURS"]
            * delta_reflectance
            * constants["COOLING_EFFICIENCY"]
            * USAGE_FACTOR
        )

        # Cap per roof (residential realism)
        annual_energy_kwh = min(
            annual_energy_kwh,
            MAX_KWH_PER_ROOF,
        )

        cost_savings = annual_energy_kwh * constants["ELECTRICITY_PRICE"]
        co2_savings = annual_energy_kwh * constants["EMISSION_FACTOR"]

        results.append({
            "label": label,
            "area_m2": area,
            "roof_type": r["type"],
            "energy_kwh_per_year": annual_energy_kwh,
            "cost_savings_per_year": cost_savings,
            "co2_savings_kg_per_year": co2_savings,
        })

    return results
