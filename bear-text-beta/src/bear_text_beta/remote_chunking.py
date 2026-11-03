"""Adapter for the isolated Chonkie service."""

from __future__ import annotations

from dataclasses import asdict
from uuid import uuid4

import httpx

from .errors import ServiceUnavailableError, ValidationError
from .models import ParsedDocument, ProcessingProfile, ProcessingResult
from .serialization import load_result


class RemoteChunkingService:
    def __init__(self, base_url: str, client: httpx.Client | None = None):
        self.client = client or httpx.Client(base_url=base_url.rstrip("/"), timeout=180)

    def process(self, document: ParsedDocument, processing_profile: ProcessingProfile | None = None,
                generation_id: str | None = None) -> ProcessingResult:
        profile = processing_profile or ProcessingProfile()
        try:
            response = self.client.post("/chunk", json={
                "document": asdict(document), "profile": asdict(profile),
                "generation_id": generation_id or str(uuid4()),
            })
            if response.status_code >= 500 or response.status_code == 429:
                raise ServiceUnavailableError("Chonkie-сервис временно недоступен")
            response.raise_for_status()
            return load_result(response.json())
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise ServiceUnavailableError("Chonkie-сервис недоступен или превысил timeout") from exc
        except (httpx.HTTPStatusError, ValueError, TypeError) as exc:
            raise ValidationError("Chonkie-сервис вернул неверный результат") from exc

    def close(self):
        self.client.close()

    def health(self) -> dict:
        try:
            response = self.client.get("/health", timeout=2)
            response.raise_for_status()
            return {"ready": response.json().get("status") == "ok"}
        except (httpx.HTTPError, ValueError, TypeError):
            return {"ready": False}
