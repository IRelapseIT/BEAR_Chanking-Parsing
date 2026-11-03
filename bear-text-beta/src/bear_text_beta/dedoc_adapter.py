"""HTTP boundary between BEAR's Python 3.12 runtime and isolated Dedoc 2.8."""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import httpx

from .errors import ParseError, ServiceUnavailableError
from .file_validation import FileLimits, ObjectStorage
from .models import Attachment, DocumentContext, ParsedDocument, SourceLocator
from .object_storage import LocalObjectStorage


class DedocClient:
    def __init__(
        self, asset_dir: Path, limits: FileLimits,
        base_url: str | None = None, client: httpx.Client | None = None,
        object_storage: ObjectStorage | None = None,
    ):
        self.limits = limits
        self.storage = object_storage or LocalObjectStorage(asset_dir)
        self.client = client or httpx.Client(
            base_url=(base_url or os.getenv("BEAR_DEDOC_URL", "http://127.0.0.1:8011")).rstrip("/"),
            timeout=httpx.Timeout(150.0, connect=5.0),
        )

    def health(self) -> dict:
        try:
            response = self.client.get("/health", timeout=2)
            response.raise_for_status()
            return {"ready": response.json().get("status") == "ok"}
        except (httpx.HTTPError, ValueError, TypeError):
            return {"ready": False}

    def close(self) -> None:
        self.client.close()

    def parse(self, data: bytes, context: DocumentContext) -> ParsedDocument:
        from .parsing import BlockBuilder

        try:
            suffix = Path(context.filename).suffix.lower()
            content_type = {
                ".txt": "text/plain", ".html": "text/html", ".htm": "text/html",
                ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            }[suffix]
            response = self.client.post(
                "/parse", files={"file": (context.filename, io.BytesIO(data),
                                        content_type)},
                data={"limits": json.dumps(vars(self.limits), sort_keys=True)},
            )
            if response.status_code >= 500 or response.status_code == 429:
                raise ServiceUnavailableError("Dedoc-сервис временно недоступен")
            response.raise_for_status()
            payload = response.json()
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise ServiceUnavailableError("Dedoc-сервис недоступен или превысил timeout") from exc
        except (httpx.HTTPStatusError, ValueError) as exc:
            raise ParseError("Dedoc не смог разобрать документ или вернул неверный ответ") from exc
        if payload.get("schema_version") != 1 or not isinstance(payload.get("blocks"), list):
            raise ParseError("Некорректный контракт ответа Dedoc")
        if len(payload["blocks"]) > self.limits.max_blocks:
            raise ParseError("Dedoc вернул слишком много блоков")
        builder = BlockBuilder(context)
        builder.warnings.extend(payload.get("warnings", []))
        attachment_by_uid = {}
        attachments = payload.get("attachments", [])
        if len(attachments) > self.limits.max_attachments:
            raise ParseError("Dedoc вернул слишком много вложений")
        total_attachment_bytes = 0
        for raw in attachments:
            try:
                uid = str(raw["uid"])
                content = base64.b64decode(raw["data_b64"], validate=True)
                if len(content) > self.limits.max_attachment_bytes:
                    raise ParseError("Вложение Dedoc превышает индивидуальный лимит")
                total_attachment_bytes += len(content)
                if total_attachment_bytes > self.limits.max_total_attachment_bytes:
                    raise ParseError("Вложения Dedoc превышают суммарный лимит")
                digest = __import__("hashlib").sha256(content).hexdigest()
                key = f"attachments/{context.tenant_id}/{context.space_id}/{digest}"
                self.storage.put_stream(key, io.BytesIO(content))
                attachment_id = str(uuid5(NAMESPACE_URL, f"{context.checksum}/{uid}"))
                builder.attachments.append(Attachment(
                    attachment_id, "image", key, raw.get("mime_type", "application/octet-stream"),
                    {"original_name": raw.get("name", ""), "checksum": digest, "stored": True},
                ))
                attachment_by_uid[uid] = attachment_id
            except (KeyError, ValueError, TypeError) as exc:
                raise ParseError("Некорректное вложение в ответе Dedoc") from exc
        text_chars = 0
        for raw in payload["blocks"]:
            try:
                kind = raw["type"]
                role = raw["role"]
                if kind not in {"TEXT", "TABLE", "IMAGE"} or not isinstance(role, str):
                    raise ValueError("invalid block kind or role")
                text = raw.get("text", "")
                if not isinstance(text, str):
                    raise ValueError("invalid text")
                text_chars += len(text)
                if text_chars > self.limits.max_text_chars:
                    raise ParseError("Dedoc вернул слишком много текста")
                page = raw.get("page")
                if page is not None and (type(page) is not int or page < 1 or page > self.limits.max_pages):
                    raise ParseError("Dedoc вернул страницу вне лимита")
                source = SourceLocator(suffix.lstrip("."), element_path=str(raw.get("node_id", "")),
                                       paragraph_index=raw.get("line_id"), page=page)
                references = raw.get("attachment_uids", [])
                if not isinstance(references, list) or any(uid not in attachment_by_uid for uid in references):
                    raise ParseError("Dedoc сослался на отсутствующее вложение")
                ids = [attachment_by_uid[uid] for uid in references]
                builder.add(text, role, source, kind=kind, level=raw.get("heading_level"),
                            metadata=raw.get("metadata", {}), attachments=ids)
            except (KeyError, TypeError, ValueError) as exc:
                raise ParseError("Некорректный блок в ответе Dedoc") from exc
        document = builder.finish()
        document.parser_profile = "dedoc-2.8-v1"
        return document
