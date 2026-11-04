"""HTTP boundary for the Python 3.12 Chonkie worker."""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from threading import Lock
from uuid import UUID

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, TypeAdapter

from .chunking import ChunkingService
from .embeddings import HFTokenCounter, InfinityEmbeddingProvider, load_profile
from .errors import BearError, ValidationError
from .models import ParsedDocument, ProcessingProfile
from .upload_limit import RequestSizeLimitMiddleware


class ChunkRequest(BaseModel):
    document: dict
    profile: dict
    generation_id: UUID


app = FastAPI(title="BEAR Chonkie Worker", version="0.2.0")
app.add_middleware(RequestSizeLimitMiddleware, max_body_bytes=100 * 1024 * 1024)
CHUNK_LOCK = Lock()


@lru_cache(maxsize=1)
def get_service():
    import os
    from pathlib import Path

    model_dir = Path(os.getenv("BEAR_MODEL_DIR", "/models/e5"))
    embedding_profile = load_profile(model_dir)
    counter = HFTokenCounter(model_dir, embedding_profile)
    provider = InfinityEmbeddingProvider(
        os.getenv("BEAR_INFINITY_URL", "http://infinity:7997/v1"), embedding_profile, counter,
        server_model=os.getenv("BEAR_INFINITY_MODEL", "bear-e5-small"),
        api_key=os.getenv("BEAR_INFINITY_API_KEY"),
    )
    return ChunkingService(counter, provider, embedding_profile)


@app.exception_handler(BearError)
def handle_error(request, exc):
    return JSONResponse(status_code=422, content={"error": type(exc).__name__, "detail": str(exc)})


@app.post("/chunk")
def chunk(request: ChunkRequest):
    try:
        document = TypeAdapter(ParsedDocument).validate_python(request.document)
        profile = ProcessingProfile(**request.profile)
    except (ValueError, TypeError) as exc:
        raise ValidationError("Некорректный документ или профиль чанкинга") from exc
    with CHUNK_LOCK:
        result = get_service().process(document, profile, str(request.generation_id))
        return asdict(result)


@app.get("/health")
def health():
    from .health import component_health
    try:
        service = get_service()
        return {"status": "ok", "chunker": "chonkie-1.7.0",
                "ready": service.counter is not None,
                "model_ready": component_health(service.provider).get("ready") is True}
    except (BearError, OSError, ValueError):
        return {"status": "unavailable", "ready": False, "model_ready": False}


@app.get("/live")
def live():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    state = health()
    return JSONResponse(status_code=200 if state["ready"] else 503, content=state)
