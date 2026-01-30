from fastapi import APIRouter
from pathlib import Path
import uuid

from backend.services.data_loader import load_geotiff
from backend.services.segmentation import RoofSegmentationService
from backend.services.postprocess import clean_roof_mask
from backend.services.reflectance import compute_reflectance_map
from backend.services.clustering import (
    extract_roof_reflectance,
    cluster_roofs_by_reflectance,
    create_thermal_cluster_mask,
)
from backend.services.energy_model import (
    compute_roof_areas,
    estimate_cooling_savings,
)
from backend.services.export import export_mask_geotiff

from backend.services.db import insert_analysis_result


router = APIRouter(prefix="/process", tags=["Process"])

MODEL_PATH = "efficientnet_unet.pth"
RESULTS_ROOT = Path("results")
RESULTS_ROOT.mkdir(exist_ok=True)

service = RoofSegmentationService(checkpoint_path=MODEL_PATH)


@router.post("/{job_id}")
def process_job(job_id: str):
    job_dir = RESULTS_ROOT / job_id
    input_path = job_dir / "input.tif"

    if not input_path.exists():
        return {"error": "Input file not found for job"}

    image, meta = load_geotiff(input_path, is_mask=False)

    raw_mask = service.predict(image)
    cleaned_mask = clean_roof_mask(raw_mask, min_area=150)

    reflectance = compute_reflectance_map(image, cleaned_mask)

    roof_stats = extract_roof_reflectance(cleaned_mask, reflectance)
    roof_stats = cluster_roofs_by_reflectance(roof_stats)
    thermal_mask = create_thermal_cluster_mask(cleaned_mask, roof_stats)

    export_mask_geotiff(raw_mask, meta, job_dir / "pred_mask.tif")
    export_mask_geotiff(cleaned_mask, meta, job_dir / "pred_mask_cleaned.tif")
    export_mask_geotiff(thermal_mask, meta, job_dir / "thermal_clusters.tif")

    transform = meta["transform"]
    pixel_area_m2 = abs(transform[0] * transform[4])

    roof_areas = compute_roof_areas(cleaned_mask, pixel_area_m2)

    constants = {
        "SOLAR_IRRADIANCE": 0.75,
        "SUNLIGHT_HOURS": 1700,
        "COOLING_EFFICIENCY": 0.65,
        "ELECTRICITY_PRICE": 0.30,
        "EMISSION_FACTOR": 0.10,
        "USAGE_FACTOR": 0.025,
        "MAX_KWH_PER_ROOF": 5000,
    }

    energy = estimate_cooling_savings(roof_stats, roof_areas, constants)

    total_energy = sum(e["energy_kwh_per_year"] for e in energy)
    total_cost = sum(e["cost_savings_per_year"] for e in energy)
    total_co2 = sum(e["co2_savings_kg_per_year"] for e in energy)

    insert_analysis_result(
    job_id=job_id,
    tile_name="input.tif",
    num_roofs=len(roof_stats),
    hot_roofs=sum(r["type"] == "hot" for r in roof_stats),
    cool_roofs=sum(r["type"] == "cool" for r in roof_stats),
    energy_kwh=total_energy,
    cost_nzd=total_cost,
    co2_kg=total_co2,
    usage_factor=constants["USAGE_FACTOR"],
    max_kwh_per_roof=constants["MAX_KWH_PER_ROOF"],
    )


    return {
        "job_id": job_id,
        "num_roofs": len(roof_stats),
        "cool_roofs": sum(r["type"] == "cool" for r in roof_stats),
        "hot_roofs": sum(r["type"] == "hot" for r in roof_stats),
        "total_energy_kwh_per_year": round(
            sum(e["energy_kwh_per_year"] for e in energy), 2
        ),
        "total_cost_nzd_per_year": round(
            sum(e["cost_savings_per_year"] for e in energy), 2
        ),
        "total_co2_kg_per_year": round(
            sum(e["co2_savings_kg_per_year"] for e in energy), 2
        ),
    }
