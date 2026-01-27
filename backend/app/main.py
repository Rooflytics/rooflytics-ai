from fastapi import FastAPI

from backend.app.routes.health import router as health_router

app = FastAPI(
    title="Rooflytics API",
    description="Urban Roof Intelligence Backend",
    version="0.1.0",
)

app.include_router(health_router)
