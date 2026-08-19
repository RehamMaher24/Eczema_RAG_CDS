from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from PIL import Image

from .schemas import ErrorResponse, HealthResponse, PredictionResponse

load_dotenv()

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png"}
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpoint = os.getenv("SKIN_MODEL_CHECKPOINT")
    if checkpoint:
        try:
            # Load the heavyweight Torch/timm stack only when a checkpoint is configured.
            from .model_service import SkinClassifier

            app.state.classifier = SkinClassifier(
                checkpoint_path=checkpoint,
                model_name=os.getenv("MODEL_NAME") or None,
                confidence_threshold=float(os.getenv("MODEL_CONFIDENCE_THRESHOLD", "0.80")),
                device=os.getenv("MODEL_DEVICE") or None,
            )
            app.state.model_error = None
        except Exception as exc:
            app.state.classifier = None
            app.state.model_error = f"Model failed to load: {type(exc).__name__}: {exc}"
    else:
        app.state.classifier = None
        app.state.model_error = "SKIN_MODEL_CHECKPOINT is not configured"
    yield


app = FastAPI(
    title="AD/CD/SD Skin Classifier API",
    version="1.0.0",
    description=(
        "Research prototype API for AD, CD, and SD image classification. "
        "Predictions are retrieval hints, not diagnoses."
    ),
    lifespan=lifespan,
)

origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {"service": "AD/CD/SD Skin Classifier API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    classifier = getattr(app.state, "classifier", None)
    if classifier is None:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            labels=[],
            message=getattr(app.state, "model_error", "Model is not loaded"),
        )
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_backbone=classifier.model_name,
        device=str(classifier.device),
        labels=list(classifier.labels),
        message="Model loaded; image output is a retrieval hint, not a diagnosis.",
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    classifier = getattr(app.state, "classifier", None)
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Classifier is not available; configure SKIN_MODEL_CHECKPOINT and restart the API.",
        )

    suffix = Path(file.filename or "").suffix.lower()
    if file.content_type not in ALLOWED_MIME_TYPES and suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail="Only JPG, JPEG, and PNG images are supported.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail=f"Image exceeds the {MAX_IMAGE_BYTES} byte limit.")

    try:
        # Decode once here so malformed files fail before model inference.
        from io import BytesIO
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid readable image.") from exc

    try:
        prediction = await run_in_threadpool(classifier.predict, image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PredictionResponse(**prediction.to_dict())
