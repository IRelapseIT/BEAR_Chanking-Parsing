import json
from contextlib import asynccontextmanager
from threading import Lock
from typing import Annotated, Literal
from uuid import NAMESPACE_URL, UUID, uuid5

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .errors import BearError, EmbeddingError, ServiceUnavailableError, StorageError
from .health import runtime_health
from .models import DocumentContext, ProcessingProfile
from .runtime import default_services
from .serialization import result_adapter
from .upload_limit import RequestSizeLimitMiddleware


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

    @app.post("/v1/jobs/{job_id}/reprocess", status_code=202)
    def reprocess_ingestion(job_id: UUID):
        new_job = get_services().manifest.reprocess_ingestion(str(job_id))
        return {"job_id": new_job, "source_job_id": str(job_id), "status": "queued"}

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

    @app.api_route("/beta/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
                   include_in_schema=False)
    def retired_protocol(path: str):
        return JSONResponse(status_code=410, content={"error": "LegacyAPIUnavailable",
            "message": "Use POST /v1/documents; client ProcessingResult is no longer accepted"})

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
