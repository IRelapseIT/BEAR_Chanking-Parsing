import json
import time
import tracemalloc
from pathlib import Path

from .errors import ValidationError
from .models import DocumentContext, ProcessingProfile


def evaluate(runtime, corpus_file: Path, modes=None):
    """Run an actual retrieval experiment; no synthetic vectors or invented scores."""
    corpus = json.loads(corpus_file.read_text(encoding="utf-8"))
    reports = []
    for mode in modes or ["adaptive", "token", "semantic"]:
        # Keep each strategy in a separate Space, leaving existing user documents intact.
        space = f"evaluation-{mode}"
        started = time.perf_counter()
        tracemalloc.start()
        counts, strategies, warnings = 0, {}, []
        inference_before = dict(getattr(runtime.embeddings, "metrics", {}))
        try:
            for entry in corpus["documents"]:
                path = (corpus_file.parent / entry["file"]).resolve()
                context = DocumentContext("evaluation", space, entry["id"], 1, path.name)
                with path.open("rb") as source:
                    data = source.read(runtime.parser.max_file_bytes + 1)
                result = runtime.process(data, context, ProcessingProfile(mode=mode))
                runtime.indexer.index(result)
                counts += len(result.chunks)
                warnings.extend(result.warnings)
                for chunk in result.chunks:
                    strategies[chunk.strategy] = strategies.get(chunk.strategy, 0) + 1
            outcomes = []
            for entry in corpus["queries"]:
                expected = set(entry["document_ids"])
                if not expected:
                    raise ValidationError("Для evaluation query нужны ожидаемые document_ids")
                hits = runtime.indexer.search(entry["query"], "evaluation", space, top_k=5)
                documents = [hit.payload["document_id"] for hit in hits]
                found = expected.intersection(documents)
                rank = next((i + 1 for i, doc in enumerate(documents) if doc in expected), None)
                outcomes.append({"query": entry["query"], "expected": sorted(expected),
                                 "retrieved": documents, "recall_at_5": len(found) / len(expected),
                                 "reciprocal_rank": 1 / rank if rank else 0})
            reports.append({
                "mode": mode, "embedding_profile_id": runtime.profile.identifier(),
                "seconds": time.perf_counter() - started,
                "python_peak_allocated_bytes": tracemalloc.get_traced_memory()[1],
                "chunks": counts, "strategies": strategies, "warnings": list(dict.fromkeys(warnings)),
                "mean_document_recall_at_5": sum(q["recall_at_5"] for q in outcomes) / max(1, len(outcomes)),
                "mean_reciprocal_rank": sum(q["reciprocal_rank"] for q in outcomes) / max(1, len(outcomes)),
                "queries": outcomes,
                "inference": {key: value - inference_before.get(key, 0)
                              for key, value in getattr(runtime.embeddings, "metrics", {}).items()},
            })
        finally:
            tracemalloc.stop()
    return {"corpus": str(corpus_file), "metric_level": "document (not exact source spans)", "reports": reports}
