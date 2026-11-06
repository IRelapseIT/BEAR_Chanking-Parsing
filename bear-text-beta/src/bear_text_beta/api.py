import json
from contextlib import asynccontextmanager
from threading import Lock
from typing import Annotated, Literal
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .errors import BearError, EmbeddingError, ServiceUnavailableError, StorageError
from .health import runtime_health
from .models import DocumentContext, ProcessingProfile
from .runtime import default_services
from .serialization import load_result, result_adapter
from .upload_limit import RequestSizeLimitMiddleware


class IndexRequest(BaseModel):
    generation_id: UUID | None = None
    result: dict | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    space_id: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)


def create_app(services=None):
    lock = Lock()
    initialization_lock = Lock()
    owned = services is None

    @asynccontextmanager
    async def lifespan(app):
        yield
        if owned and app.state.services is not None:
            app.state.services.close()

    app = FastAPI(title="BEAR Text Beta", version="0.2.0", lifespan=lifespan)
    app.add_middleware(RequestSizeLimitMiddleware, max_body_bytes=51 * 1024 * 1024)
    app.state.services = services

    def get_services():
        if app.state.services is None:
            with initialization_lock:
                if app.state.services is None:
                    app.state.services = default_services()
        return app.state.services

    @app.post("/v1/documents", status_code=202)
    async def submit_document(file: Annotated[UploadFile, File()], context: Annotated[str, Form()]):
        def enqueue():
            from dataclasses import asdict

            from pydantic import TypeAdapter

            from .errors import ValidationError
            try:
                ctx = TypeAdapter(DocumentContext).validate_python(json.loads(context))
            except (ValueError, TypeError) as exc:
                raise ValidationError("Некорректный DocumentContext") from exc
            if ctx.document_version < 1 or any(not value.strip() for value in (
                    ctx.tenant_id, ctx.space_id, ctx.document_id, ctx.filename)) or file.filename != ctx.filename:
                raise ValidationError("Недопустимый контекст или имя файла")
            with lock:
                runtime = get_services()
            stored = runtime.upload_stream(file.file, ctx)
            ctx.checksum = stored.checksum
            profile = ProcessingProfile()
            identity = json.dumps([asdict(ctx), asdict(profile), runtime.profile.identifier()], sort_keys=True)
            job_id = str(uuid5(NAMESPACE_URL, identity))
            try:
                created = runtime.manifest.submit_ingestion(job_id, stored, ctx, profile)
            except Exception:
                runtime.object_storage.delete(stored.object_key)
                raise
            if not created:
                runtime.object_storage.delete(stored.object_key)
            return {"job_id": job_id, "status": "queued" if created else "already_submitted",
                    "status_url": f"/v1/jobs/{job_id}"}
        return await run_in_threadpool(enqueue)

    @app.get("/v1/jobs/{job_id}")
    def ingestion_status(job_id: UUID):
        runtime = get_services()
        state = runtime.manifest.processing_status(str(job_id))
        if state is None:
            return JSONResponse(status_code=404, content={"error": "JobNotFound"})
        state["worker_status"] = state["status"]
        state["status"] = ("failed" if state["status"] == "failed" else state["phase"])
        try:
            result = runtime.manifest.load(str(job_id))
            state["acceptance"] = result.metrics.get("acceptance")
            state["chunks"] = len(result.chunks)
            state["warnings"] = result.warnings
        except StorageError:
            pass
        return state

    @app.get("/v1/jobs/{job_id}/result")
    def ingestion_result(job_id: UUID):
        result = get_services().manifest.load(str(job_id))
        return result_adapter().dump_python(result, mode="json")

    @app.post("/v1/jobs/{job_id}/retry", status_code=202)
    def retry_ingestion(job_id: UUID):
        get_services().manifest.retry_ingestion(str(job_id))
        return {"job_id": str(job_id), "status": "queued"}

    @app.delete("/v1/documents/{document_id}")
    def delete_document(document_id: str, tenant_id: str, space_id: str):
        runtime = get_services()
        generations = runtime.manifest.delete_document(tenant_id, space_id, document_id)
        pending = []
        for generation in generations:
            try:
                runtime.vectors.delete_generation(generation)
            except StorageError:
                pending.append(generation)
        return {"status": "deleted", "cleanup_pending": pending}

    @app.post("/v1/search")
    def search_documents(request: SearchRequest):
        hits = get_services().indexer.search(request.query, request.tenant_id, request.space_id, request.top_k)
        return {"results": [{"chunk_id": hit.id, "score": hit.score, "text": hit.payload["text"],
                             "document_id": hit.payload["document_id"],
                             "document_version": hit.payload["document_version"],
                             "section_path": hit.payload.get("section_path", []),
                             "sources": hit.payload.get("sources", []),
                             "acceptance": hit.payload.get("acceptance", {}),
                             "generation_id": hit.payload["generation_id"]} for hit in hits]}

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
                return runtime.indexer.retry(str(request.generation_id))
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
                return runtime_health(get_services())
            except (BearError, OSError, ValueError):
                return {"status": "ok", "model_ready": False,
                        "capabilities": {key: False for key in ("parse", "chunk", "index", "search")}}

    @app.get("/live")
    def live():
        return {"status": "ok"}

    @app.get("/ready")
    def ready(capability: Literal["parse", "chunk", "index", "search"] = "index"):
        state = health()
        available = state["capabilities"].get(capability, False)
        return JSONResponse(status_code=200 if available else 503,
                            content={**state, "ready": available, "capability": capability})

    return app


app = create_app()
