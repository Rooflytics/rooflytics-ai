from fastapi import APIRouter, UploadFile, File
from pathlib import Path
import uuid

router = APIRouter(prefix="/upload", tags=["Upload"])

RESULTS_ROOT = Path("results")
RESULTS_ROOT.mkdir(exist_ok=True)


@router.post("/")
async def upload_geotiff(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    job_dir = RESULTS_ROOT / job_id
    job_dir.mkdir()

    input_path = job_dir / "input.tif"

    with open(input_path, "wb") as f:
        f.write(await file.read())

    return {
        "job_id": job_id,
        "message": "File uploaded successfully",
    }
