from __future__ import annotations

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .dependencies import get_service, get_settings
from .schemas import ChatResponse, RetrieveRequest, RetrieveResponse
from .services import ClinicalRagService, ServiceError

settings = get_settings()
app = FastAPI(title="Eczema Clinical RAG Prototype", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_credentials=False, allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(ServiceError)
async def service_error_handler(_, exc: ServiceError):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.get("/health")
def health(service: ClinicalRagService = Depends(get_service)) -> dict[str, object]:
    return service.health()


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(payload: RetrieveRequest, service: ClinicalRagService = Depends(get_service)) -> RetrieveResponse:
    scope, routing, evidence, timings = service.retrieve(payload.question, payload.top_k, payload.document_filters)
    return RetrieveResponse(question=payload.question.strip(), scope_check=scope, routing=routing, evidence=evidence, timings_ms=timings)


@app.post("/chat", response_model=ChatResponse)
async def chat(question: str = Form(...), image: UploadFile | None = File(None), top_k: int | None = Form(None), service: ClinicalRagService = Depends(get_service)) -> ChatResponse:
    if top_k is not None and not 1 <= top_k <= 20:
        raise HTTPException(status_code=422, detail="top_k must be between 1 and 20.")
    image_bytes = await image.read() if image else None
    payload = service.chat(question, image_bytes, image.filename if image else "", image.content_type if image else None, top_k)
    return ChatResponse.model_validate(payload)
