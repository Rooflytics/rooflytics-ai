from fastapi import FastAPI

from backend.app.routes.health import router as health_router
from backend.app.routes.upload import router as upload_router
from backend.app.routes.process import router as process_router
from backend.app.routes.results import router as results_router

app = FastAPI(
    title="Rooflytics API",
    description="Urban Roof Intelligence Backend",
    version="0.1.0",
)

app.include_router(health_router)
app.include_router(upload_router)
app.include_router(process_router)
app.include_router(results_router)