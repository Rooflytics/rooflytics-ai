from pathlib import Path
from backend.services.data_loader import load_geotiff
from backend.services.segmentation import RoofSegmentationService
from backend.services.postprocess import clean_roof_mask
from backend.services.reflectance import compute_reflectance_map
from backend.utils.logging import get_logger

HERE = Path(__file__).parent
OUTPUT_DIR = HERE / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

image, meta = load_geotiff(
    HERE / "christchurch_41.tif",
    is_mask=False,
)

service = RoofSegmentationService(
    checkpoint_path="efficientnet_unet.pth"
)

raw_mask = service.predict(image)
cleaned_mask = clean_roof_mask(raw_mask, min_area=150)

reflectance = compute_reflectance_map(
    image=image,
    roof_mask=cleaned_mask,
)

print(
    "Reflectance stats:",
    reflectance[reflectance > 0].min(),
    reflectance[reflectance > 0].mean(),
    reflectance[reflectance > 0].max(),
)

logger = get_logger("Pipeline")

logger.info(f"Raw mask positive %: {raw_mask.mean() * 100:.3f}")
logger.info(f"Cleaned mask positive %: {cleaned_mask.mean() * 100:.3f}")

logger.info(
    f"Reflectance stats — "
    f"min={reflectance[reflectance > 0].min():.5f}, "
    f"mean={reflectance[reflectance > 0].mean():.5f}, "
    f"max={reflectance[reflectance > 0].max():.5f}"
)

from backend.services.clustering import (
    extract_roof_reflectance,
    cluster_roofs_by_reflectance,
)

roof_stats = extract_roof_reflectance(
    roof_mask=cleaned_mask,
    reflectance_map=reflectance,
)

roof_stats = cluster_roofs_by_reflectance(roof_stats)

cool = sum(1 for r in roof_stats if r["type"] == "cool")
hot = sum(1 for r in roof_stats if r["type"] == "hot")

print(f"Total roofs: {len(roof_stats)}")
print(f"Cool roofs: {cool}")
print(f"Hot roofs: {hot}")

# Inspect a few
for r in roof_stats[:5]:
    print(r)

from backend.services.export import export_mask_geotiff

export_mask_geotiff(
    raw_mask,
    meta,
    OUTPUT_DIR / "christchurch_41_pred_raw.tif",
)

export_mask_geotiff(
    cleaned_mask,
    meta,
    OUTPUT_DIR / "christchurch_41_pred_cleaned.tif",
)


from backend.services.clustering import create_thermal_cluster_mask

thermal_mask = create_thermal_cluster_mask(
    roof_mask=cleaned_mask,
    roof_stats=roof_stats,
)

export_mask_geotiff(
    thermal_mask,
    meta,
    OUTPUT_DIR / "christchurch_41_thermal_clusters.tif",
)

print("Saved thermal cluster map")

from backend.services.energy_model import (
    compute_roof_areas,
    estimate_cooling_savings,
)

# Pixel area from GeoTIFF
transform = meta["transform"]
pixel_area_m2 = abs(transform[0] * transform[4])

roof_areas = compute_roof_areas(
    cleaned_mask,
    pixel_area_m2,
)

constants = {
    "SOLAR_IRRADIANCE": 0.75,
    "SUNLIGHT_HOURS": 1700,
    "COOLING_EFFICIENCY": 0.65,
    "ELECTRICITY_PRICE": 0.30,
    "EMISSION_FACTOR": 0.10,
}

energy_results = estimate_cooling_savings(
    roof_stats,
    roof_areas,
    constants,
)

# Summarize
total_energy = sum(r["energy_kwh_per_year"] for r in energy_results)
total_cost = sum(r["cost_savings_per_year"] for r in energy_results)
total_co2 = sum(r["co2_savings_kg_per_year"] for r in energy_results)

print("Total roofs:", len(energy_results))
print(f"Total energy savings: {total_energy:.2f} kWh/year")
print(f"Total cost savings: NZD {total_cost:.2f}/year")
print(f"Total CO₂ reduction: {total_co2:.2f} kg/year")
