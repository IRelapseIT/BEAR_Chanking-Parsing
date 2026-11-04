"""Adapter for the isolated Chonkie service."""

from __future__ import annotations

import json
from dataclasses import asdict
from uuid import UUID, uuid4

import httpx

from .errors import ServiceUnavailableError, ValidationError
from .models import EmbeddingProfile, ParsedDocument, ProcessingProfile, ProcessingResult
from .serialization import load_result


class RemoteChunkingService:
    def __init__(self, base_url: str, client: httpx.Client | None = None, *,
                 counter=None, embedding_profile: EmbeddingProfile | None = None,
                 external_validators=None, max_response_bytes: int = 64 * 1024 * 1024):
        if isinstance(max_response_bytes, bool) or not isinstance(max_response_bytes, int) or max_response_bytes < 1:
            raise ValueError("max_response_bytes must be a positive integer")
        self.client = client or httpx.Client(base_url=base_url.rstrip("/"), timeout=180)
        self.counter = counter
        self.embedding_profile = embedding_profile
        self.external_validators = dict(external_validators or {})
        self.max_response_bytes = max_response_bytes

    @staticmethod
    def _canonical(data):
        return json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))

    def process(self, document: ParsedDocument, processing_profile: ProcessingProfile | None = None,
                generation_id: str | None = None) -> ProcessingResult:
        profile = processing_profile or ProcessingProfile()
        if self.counter is None or self.embedding_profile is None:
            raise ValidationError("Remote chunker требует локальные tokenizer и embedding profile")
        from .chunking import ChunkingService, ChunkValidator

        # Keep the exact immutable wire request, including generated ids, for comparison.
        generation = str(UUID(generation_id)) if generation_id else str(uuid4())
        document_payload = json.loads(self._canonical(asdict(document)))
        expected_profile_id = ChunkingService(
            self.counter, None, self.embedding_profile).profile_id(profile, document.parser_profile)
        try:
            with self.client.stream("POST", "/chunk", json={
                "document": document_payload, "profile": asdict(profile), "generation_id": generation,
            }) as response:
                if response.status_code >= 500 or response.status_code == 429:
                    raise ServiceUnavailableError("Chonkie-сервис временно недоступен")
                response.raise_for_status()
                length = response.headers.get("content-length")
                if length is not None and int(length) > self.max_response_bytes:
                    raise ValidationError("Ответ Chonkie-сервиса превышает лимит")
                body = bytearray()
                for part in response.iter_bytes():
                    if len(body) + len(part) > self.max_response_bytes:
                        raise ValidationError("Ответ Chonkie-сервиса превышает лимит")
                    body.extend(part)
            result = load_result(json.loads(body))
            if (self._canonical(asdict(result.parsed_document)) != self._canonical(document_payload)
                    or self._canonical(asdict(result.context)) != self._canonical(document_payload["context"])
                    or result.generation_id != generation):
                raise ValidationError("Ответ Chonkie-сервиса не соответствует исходному документу")
            if (result.processing_profile_id != expected_profile_id
                    or result.metrics.get("processing_profile") != asdict(profile)
                    or result.metrics.get("embedding_profile") != asdict(self.embedding_profile)
                    or result.metrics.get("embedding_profile_id") != self.embedding_profile.identifier()
                    or result.metrics.get("token_counter") != type(self.counter).__name__):
                raise ValidationError("Ответ Chonkie-сервиса создан с несовместимым профилем")
            ChunkValidator(self.counter, min(profile.max_tokens, self.embedding_profile.max_input_tokens),
                           self.external_validators).validate(result)
            return result
        except httpx.RequestError as exc:
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
