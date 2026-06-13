from pathlib import Path

from fastapi import FastAPI

from app.core.config import settings
from app.routers import fit, grade
from app.schemas.passport import HealthResponse


app = FastAPI(
    title="Relay ML Service",
    description="Image/video grading and fit intelligence service for Relay.",
    version=settings.app_version,
)


def _model_status(model_path: Path) -> tuple[bool, int, list[str]]:
    notes: list[str] = []

    if not model_path.exists():
        notes.append("CNN model file is missing. Phase 1 can run, but grading will use stubs.")
        return False, 0, notes

    model_bytes = model_path.stat().st_size
    if model_bytes == 0:
        notes.append("CNN model file exists but is empty; replace it with trained weights in Phase 3.")
        return False, model_bytes, notes

    return True, model_bytes, notes


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    model_loaded, model_bytes, notes = _model_status(settings.cnn_model_path)
    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        cnn_version="v1",
        model_path=str(settings.cnn_model_path),
        model_bytes=model_bytes,
        notes=notes,
    )


app.include_router(grade.router)
app.include_router(fit.router)
