import json
from contextlib import asynccontextmanager
from threading import Lock
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .errors import BearError, EmbeddingError, ServiceUnavailableError, StorageError
from .models import DocumentContext, ProcessingProfile
from .runtime import default_services
from .serialization import load_result, result_adapter
from .upload_limit import RequestSizeLimitMiddleware


class IndexRequest(BaseModel):
    generation_id: str | None = None
    result: dict | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    space_id: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)


def create_app(services=None):
    lock = Lock()
    owned = services is None

    @asynccontextmanager
    async def lifespan(app):
        yield
        if owned and app.state.services is not None:
            app.state.services.close()

    app = FastAPI(title="BEAR Text Beta", version="0.1.0", lifespan=lifespan)
    app.add_middleware(RequestSizeLimitMiddleware, max_body_bytes=51 * 1024 * 1024)
    app.state.services = services

    def get_services():
        if app.state.services is None:
            app.state.services = default_services()
        return app.state.services

    @app.exception_handler(BearError)
    async def expected_error(request, exc):
        status = 503 if isinstance(exc, (EmbeddingError, StorageError, ServiceUnavailableError)) else 422
        return JSONResponse(status_code=status, content={"error": type(exc).__name__, "detail": str(exc)})

    @app.post("/beta/process")
    async def process(file: Annotated[UploadFile, File()], context: Annotated[str, Form()]):
        def work():
            from pydantic import TypeAdapter

            from .errors import ValidationError
            try:
                ctx = TypeAdapter(DocumentContext).validate_python(json.loads(context))
            except (ValueError, TypeError) as exc:
                raise ValidationError("Некорректный JSON DocumentContext") from exc
            with lock:
                runtime = get_services()
                stored = runtime.upload_stream(file.file, ctx)
                if stored.size_bytes > 8 * 1024 * 1024:
                    job_id = str(uuid4())
                    runtime.manifest.submit_processing(job_id, stored, ctx, ProcessingProfile())
                    return JSONResponse(status_code=202, content={
                        "job_id": job_id, "status": "pending", "size_bytes": stored.size_bytes,
                    })
                result = runtime.process_stored(stored, ctx, ProcessingProfile())
                return result_adapter().dump_python(result, mode="json")
        return await run_in_threadpool(work)

    @app.get("/beta/jobs/{job_id}")
    def job_status(job_id: str):
        with lock:
            state = get_services().manifest.processing_status(job_id)
        if state is None:
            return JSONResponse(status_code=404, content={"error": "JobNotFound"})
        return state

    @app.post("/beta/index")
    def index(request: IndexRequest):
        from .errors import ValidationError
        with lock:
            runtime = get_services()
            if bool(request.generation_id) == bool(request.result):
                raise ValidationError("Укажите generation_id либо result")
            if request.generation_id:
                return runtime.indexer.retry(request.generation_id)
            try:
                result = load_result(request.result)
            except (ValueError, TypeError) as exc:
                raise ValidationError("Некорректный ProcessingResult") from exc
            return runtime.indexer.index(result)

    @app.post("/beta/search")
    def search(request: SearchRequest):
        with lock:
            hits = get_services().indexer.search(request.query, request.tenant_id, request.space_id, request.top_k)
        return {"results": [{"chunk_id": hit.id, "score": hit.score, "text": hit.payload["text"],
                             "source_spans": hit.payload.get("source_spans", []), "payload": hit.payload}
                            for hit in hits]}

    @app.get("/health")
    def health():
        with lock:
            try:
                runtime = get_services()
                dedoc = getattr(getattr(runtime, "parser", None), "parsers", {}).get(".docx")
                chunker = getattr(runtime, "chunker", None)
                return {"status": "ok", "model_ready": True,
                        "infinity": runtime.embeddings.health(), "qdrant": runtime.vectors.health(),
                        "dedoc": dedoc.health() if hasattr(dedoc, "health") else {"ready": False},
                        "chunker": chunker.health() if hasattr(chunker, "health")
                        else {"ready": True, "mode": "in-process"}}
            except BearError as exc:
                return {"status": "ok", "model_ready": False, "detail": str(exc)}

    return app


app = create_app()
