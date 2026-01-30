from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

router = APIRouter(prefix="/results", tags=["Results"])

RESULTS_ROOT = Path("results")


@router.get("/{job_id}")
def list_results(job_id: str):
    job_dir = RESULTS_ROOT / job_id

    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    files = [
        f.name for f in job_dir.iterdir()
        if f.is_file()
    ]

    return {
        "job_id": job_id,
        "files": files,
    }


@router.get("/{job_id}/{filename}")
def download_result(job_id: str, filename: str):
    file_path = RESULTS_ROOT / job_id / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream",
    )
