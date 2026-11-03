import math
import time
from pathlib import Path

import httpx
import numpy as np
from chonkie.embeddings import BaseEmbeddings

from .errors import EmbeddingError, ValidationError
from .models import EmbeddingProfile


class HFTokenCounter:
    def __init__(self, model_path: Path, profile: EmbeddingProfile):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True,
                                                       use_fast=True, trust_remote_code=False)
        self.profile = profile

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False, truncation=False))

    def full_count(self, text: str, purpose: str = "document") -> int:
        return len(self.tokenizer.encode(self.profile.prefix(purpose) + text,
                                         add_special_tokens=True, truncation=False))


def validate_vectors(vectors, expected_count: int, dimension: int):
    if not isinstance(vectors, list) or len(vectors) != expected_count:
        raise EmbeddingError("Количество vectors не соответствует входным текстам")
    for vector in vectors:
        if (not isinstance(vector, (list, np.ndarray)) or len(vector) != dimension
                or any(isinstance(v, bool) or not isinstance(v, (int, float, np.number))
                       or not math.isfinite(float(v)) for v in vector)
                or sum(float(v) ** 2 for v in vector) <= 0):
            raise EmbeddingError("Невалидный embedding: размерность, NaN/Inf или нулевой vector")


class InfinityEmbeddingProvider:
    def __init__(self, base_url, profile, counter, *, server_model=None, max_tokens=480,
                 client=None, api_key=None):
        self.profile = profile
        self.counter = counter
        self.max_tokens = min(max_tokens, profile.max_input_tokens)
        self.server_model = server_model or profile.model_id
        self.client = client or httpx.Client(base_url=base_url.rstrip("/") + "/", timeout=30,
                                            headers={"Authorization": f"Bearer {api_key}"} if api_key else {})
        self.metrics = {"requests": 0, "input_texts": 0, "input_tokens": 0}

    def close(self):
        self.client.close()

    def _request(self, inputs):
        response = None
        for attempt in range(2):
            try:
                self.metrics["requests"] += 1
                response = self.client.post("embeddings", json={"model": self.server_model, "input": inputs})
                if response.status_code in {408, 429, 500, 502, 503, 504} and attempt == 0:
                    time.sleep(0.1)
                    continue
                response.raise_for_status()
                body = response.json()
                entries = body["data"]
                if not isinstance(entries, list):
                    raise ValueError("Invalid data array")
                indices = [entry["index"] for entry in entries]
                if any(type(i) is not int for i in indices) or sorted(indices) != list(range(len(inputs))):
                    raise ValueError("Invalid embedding indices")
                vectors = [entry["embedding"] for entry in sorted(entries, key=lambda item: item["index"])]
                validate_vectors(vectors, len(inputs), self.profile.vector_dimension)
                return vectors
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt == 0:
                    time.sleep(0.1)
                    continue
                raise EmbeddingError("Infinity недоступен или превысил timeout") from exc
            except (httpx.HTTPStatusError, ValueError, KeyError, TypeError) as exc:
                raise EmbeddingError("Некорректный ответ Infinity") from exc
        raise EmbeddingError(f"Infinity failure: {response}")

    def embed_batch(self, texts, purpose):
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise EmbeddingError("Пустой вход embedding-модели")
        counts = [self.counter.full_count(text, purpose) for text in texts]
        if any(count > self.max_tokens for count in counts):
            raise EmbeddingError("Вход Infinity превышает токенный бюджет; усечение запрещено")
        vectors = []
        prefix = self.profile.prefix(purpose)
        for index in range(0, len(texts), 16):
            batch = texts[index:index + 16]
            self.metrics["input_texts"] += len(batch)
            self.metrics["input_tokens"] += sum(counts[index:index + 16])
            vectors.extend(self._request([prefix + text for text in batch]))
        return vectors

    def health(self):
        try:
            response = self.client.get("models")
            response.raise_for_status()
            models = response.json()["data"]
            return {"ready": any(model.get("id") == self.server_model for model in models)}
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return {"ready": False}


class InfinityChonkieEmbeddings(BaseEmbeddings):
    def __init__(self, provider, counter, profile):
        super().__init__()
        self.provider = provider
        self.counter = counter
        self.profile = profile

    @property
    def dimension(self):
        return self.profile.vector_dimension

    def embed(self, text):
        return self.embed_batch([text])[0]

    def embed_batch(self, texts):
        vectors = self.provider.embed_batch(texts, "similarity")
        validate_vectors(vectors, len(texts), self.dimension)
        return [np.asarray(vector, dtype=np.float32) for vector in vectors]

    def get_tokenizer(self):
        return self.counter.tokenizer


class UnavailableEmbeddingProvider:
    def embed_batch(self, texts, purpose):
        raise EmbeddingError("Инференс отключён: используется резервный чанкинг")

    def health(self):
        return {"ready": False, "disabled": True}


class CharacterTokenCounter:
    """Explicit test-only counter; its results may never be indexed with a real model."""
    def __init__(self, profile):
        from chonkie.tokenizer import CharacterTokenizer
        self.profile = profile
        self.tokenizer = CharacterTokenizer()

    def count(self, text):
        return len(text)

    def full_count(self, text, purpose="document"):
        return len(self.profile.prefix(purpose) + text) + 2


def load_profile(model_path: Path):
    import json
    manifest_path = model_path / "bear-model.json"
    if not manifest_path.exists():
        raise ValidationError("Сначала выполните bear prepare-model; runtime не скачивает модель")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    profile = EmbeddingProfile(**manifest["embedding_profile"])
    if not profile.model_revision or not profile.tokenizer_revision:
        raise ValidationError("Не зафиксирована ревизия модели")
    # Verify the local tokenizer has not been silently replaced after preparation.
    import hashlib
    for filename, expected in manifest["tokenizer_checksums"].items():
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts or (model_path / relative).resolve().is_relative_to(
                model_path.resolve()) is False:
            raise ValidationError("Invalid tokenizer manifest path")
        actual = hashlib.sha256((model_path / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValidationError(f"Изменён файл токенизатора: {filename}")
    return profile
