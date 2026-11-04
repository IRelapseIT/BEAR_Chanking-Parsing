"""Liveness and capability readiness have different dependency requirements."""

import httpx

from .errors import BearError


def component_health(component):
    if component is None or not callable(getattr(component, "health", None)):
        return {"ready": False}
    try:
        value = component.health()
        return value if isinstance(value, dict) else {"ready": False}
    except (BearError, httpx.HTTPError, OSError, ValueError, TypeError):
        return {"ready": False}


def runtime_health(runtime):
    parser = getattr(runtime, "parser", None)
    counter_ready = getattr(runtime, "counter", None) is not None
    infinity = component_health(getattr(runtime, "embeddings", None))
    qdrant = component_health(getattr(runtime, "vectors", None))
    dedoc = component_health(getattr(parser, "parsers", {}).get(".docx"))
    chunker = getattr(runtime, "chunker", None)
    chunk_health = (component_health(chunker) if callable(getattr(chunker, "health", None))
                    else {"ready": chunker is not None, "mode": "in-process"})
    model_ready = infinity.get("ready") is True
    chunk_ready = counter_ready and chunk_health.get("ready") is True
    index_ready = chunk_ready and model_ready and qdrant.get("ready") is True
    return {"status": "ok", "model_ready": model_ready, "infinity": infinity,
            "qdrant": qdrant, "dedoc": dedoc, "chunker": chunk_health,
            "capabilities": {"parse": parser is not None, "chunk": chunk_ready,
                             "index": index_ready, "search": index_ready}}
