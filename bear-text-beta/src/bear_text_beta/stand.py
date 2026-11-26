"""Local stand views over the canonical ingestion and search services."""
import io
import json
from dataclasses import asdict
from pathlib import Path
from typing import Literal
from urllib.parse import quote
from uuid import NAMESPACE_URL, UUID, uuid5

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field, TypeAdapter

from .chunking import external_fallback_text
from .comparison import compare_results
from .errors import StorageError, ValidationError
from .file_validation import StoredObject
from .models import DocumentContext, ProcessingProfile


class RechunkRequest(BaseModel):
    mode: Literal["adaptive", "fixed", "token", "semantic"] = "adaptive"
    target_tokens: int = Field(default=320, ge=32, le=8192)
    max_tokens: int = Field(default=480, ge=32, le=8192)
    threshold: float = Field(default=0.8, gt=0, lt=1)
    similarity_window: int = Field(default=2, ge=1, le=16)
    fixed_characters: int = Field(default=1000, ge=8, le=100_000)


def paragraphs(document):
    for block in document.blocks:
        text = block.text or external_fallback_text(block)
        if text.strip():
            yield text


def document_text(document, markdown=False):
    for block in document.blocks:
        text = block.text or external_fallback_text(block)
        if not text.strip():
            continue
        if markdown:
            if block.role == "heading":
                text = "#" * min(6, max(1, len(block.section_path))) + " " + text
            elif block.role == "list_item":
                text = "- " + text
            elif block.role == "code":
                fence = "~" * max(3, max((len(part) for part in __import__("re").findall(r"~+", text)), default=0) + 1)
                text = f"{fence}\n{text}\n{fence}"
        yield text + "\n\n"


def download(filename):
    return {"Content-Disposition": "attachment; filename*=UTF-8''" + quote(filename, safe=""),
            "X-Content-Type-Options": "nosniff"}


def stream_object(storage, key):
    with storage.get_stream(key) as source:
        yield from iter(lambda: source.read(1024 * 1024), b"")


def raster_page(storage, key, filename, page, rotation):
    from PIL import Image, ImageOps

    with storage.get_stream(key) as original:
        if Path(filename).suffix.lower() == ".pdf":
            import pypdfium2 as pdfium

            with pdfium.PdfDocument(original.read()) as document:
                if not 1 <= page <= len(document):
                    raise HTTPException(404, "Страница не найдена")
                with document[page - 1] as pdf_page:
                    width, height = pdf_page.get_size()
                    if not width or not height:
                        raise HTTPException(422, "Некорректный размер страницы")
                    scale = min(2.0, 1800 / max(width, height))
                    with pdf_page.render(scale=scale, rotation=rotation) as bitmap:
                        image = bitmap.to_pil().copy()
                count = len(document)
        else:
            if page != 1:
                raise HTTPException(404, "Страница не найдена")
            with Image.open(original) as source:
                if source.width * source.height > 25_000_000:
                    raise HTTPException(422, "Слишком большое изображение")
                image = ImageOps.exif_transpose(source).convert("RGB").rotate(-rotation, expand=True)
                image.thumbnail((1800, 1800))
            count = 1
    output = io.BytesIO()
    image.convert("RGB").save(output, format="JPEG", quality=88)
    return Response(output.getvalue(), media_type="image/jpeg", headers={"X-Page-Count": str(count)})


def stand_router(get_services):
    router = APIRouter(prefix="/v1")

    def scoped(job_id, tenant_id, space_id):
        runtime = get_services()
        row = runtime.manifest.job_record(str(job_id), tenant_id, space_id)
        if row is None:
            raise HTTPException(404, "Прогон не найден в выбранной области")
        return runtime, row

    @router.get("/formats")
    def formats():
        runtime = get_services()
        return {"extensions": sorted(runtime.parser.parsers),
                "embedding_profile": asdict(runtime.profile), "profile_id": runtime.profile.identifier(),
                "collection": getattr(runtime.vectors, "collection", ""),
                "strategies": ["adaptive", "fixed", "token", "semantic"],
                "defaults": asdict(ProcessingProfile())}

    @router.get("/jobs")
    def list_jobs(tenant_id: str, space_id: str, offset: int = Query(0, ge=0),
                  limit: int = Query(50, ge=1, le=200)):
        jobs = get_services().manifest.list_jobs(tenant_id, space_id, limit, offset)
        for job in jobs:
            del job["stored"]
        return {"jobs": jobs, "offset": offset}

    @router.post("/jobs/{job_id}/rechunk", status_code=202)
    def rechunk(job_id: UUID, request: RechunkRequest, tenant_id: str, space_id: str):
        runtime, row = scoped(job_id, tenant_id, space_id)
        runtime.manifest.load_parsed(str(job_id))
        values = request.model_dump()
        try:
            profile = ProcessingProfile(semantic_region_tokens=max(2048, values["max_tokens"]), **values)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        if profile.max_tokens > runtime.profile.max_input_tokens:
            raise ValidationError("Лимит превышает размер входа embedding-модели")
        identity = json.dumps([str(job_id), asdict(profile), runtime.profile.identifier()], sort_keys=True)
        new_id = str(uuid5(NAMESPACE_URL, identity))
        created = runtime.manifest.submit_ingestion(
            new_id, StoredObject(**row["stored"]), DocumentContext(**row["context"]), profile, parse_from=str(job_id))
        return {"job_id": new_id, "status": "queued" if created else "already_submitted"}

    @router.get("/jobs/{job_id}/compare/{other_id}")
    def compare(job_id: UUID, other_id: UUID, tenant_id: str, space_id: str):
        runtime, _ = scoped(job_id, tenant_id, space_id)
        scoped(other_id, tenant_id, space_id)
        return compare_results(runtime.manifest.load(str(job_id)), runtime.manifest.load(str(other_id)))

    @router.get("/jobs/{job_id}/page/{page}")
    def preview(job_id: UUID, page: int, tenant_id: str, space_id: str,
                rotation: Literal[0, 90, 180, 270] = 0):
        runtime, row = scoped(job_id, tenant_id, space_id)
        filename = row["context"]["filename"]
        if Path(filename).suffix.lower() not in {".pdf", ".png", ".jpg", ".jpeg"}:
            raise HTTPException(415, "Для данного формата доступен структурный просмотр")
        try:
            return raster_page(runtime.object_storage, row["stored"]["object_key"], filename, page, rotation)
        except (ValueError, RuntimeError, OSError) as exc:
            raise HTTPException(422, "Не удалось построить превью страницы") from exc

    @router.get("/jobs/{job_id}/artifacts/{kind}")
    def artifact(job_id: UUID, kind: Literal["original", "parsed_document.json", "document.md", "document.txt",
                  "chunks.jsonl", "warnings.json", "configuration.json", "report.md"],
                 tenant_id: str, space_id: str, inline: bool = False):
        runtime, row = scoped(job_id, tenant_id, space_id)
        if kind == "original":
            return StreamingResponse(stream_object(runtime.object_storage, row["stored"]["object_key"]),
                                     media_type="application/octet-stream",
                                     headers=download(row["context"]["filename"]))
        try:
            result = runtime.manifest.load(str(job_id))
        except StorageError:
            result = None
        document, parse_seconds = runtime.manifest.load_parsed(str(job_id))
        headers = {} if inline else download(kind)
        if kind == "parsed_document.json":
            return Response(TypeAdapter(type(document)).dump_json(document), media_type=application_json, headers=headers)
        if kind in {"document.md", "document.txt"}:
            return StreamingResponse(document_text(document, kind.endswith(".md")),
                                     media_type="text/plain; charset=utf-8", headers=headers)
        if kind == "chunks.jsonl":
            chunks = result.chunks if result else ()
            return StreamingResponse((json.dumps(asdict(chunk), ensure_ascii=False) + "\n" for chunk in chunks),
                                     media_type="application/x-ndjson", headers=headers)
        metrics = result.metrics if result else {}
        if kind == "warnings.json":
            value = {"parser": document.warnings, "chunking": result.warnings if result else [],
                     "quality": metrics.get("acceptance", {}).get("warnings", [])}
        elif kind == "configuration.json":
            value = {"parsing_profile": document.parser_profile, "chunking": row["profile"],
                     "embedding": metrics.get("embedding_profile") or asdict(runtime.profile)}
        else:
            lines = [f"# Прогон {job_id}", "", f"- Блоков: {len(document.blocks)}",
                     f"- Чанков: {len(result.chunks) if result else 0}",
                     f"- Парсинг, с: {parse_seconds}",
                     f"- Индекс: {runtime.manifest.index_status(str(job_id))['status']}", "",
                     "Покрытие извлечённого текста не определяет полноту оригинала."]
            return Response("\n".join(lines), media_type="text/plain", headers=headers)
        return Response(json.dumps(value, ensure_ascii=False), media_type=application_json, headers=headers)

    return router


application_json = "application/json"
