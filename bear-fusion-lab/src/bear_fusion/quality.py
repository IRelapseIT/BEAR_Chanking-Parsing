"""Transparent local lexical diagnostic; not vector retrieval evaluation."""
import math
import re
from collections import Counter


def search(chunks, query, top_k=5):
    tokens = lambda text: re.findall(r"\w+", text.lower())
    terms = tokens(query)
    docs = [Counter(tokens(c["text"])) for c in chunks]
    average = sum(sum(d.values()) for d in docs) / max(len(docs), 1)
    scores = []
    for chunk, doc in zip(chunks, docs, strict=True):
        score = 0.0
        for term in terms:
            df = sum(term in d for d in docs)
            frequency = doc[term]
            score += math.log(1 + (len(docs) - df + .5) / (df + .5)) * frequency * 2.2 / (
                frequency + 1.2 * (.25 + .75 * sum(doc.values()) / max(average, 1)))
        if score:
            scores.append({"score": score, "chunk": chunk})
    return sorted(scores, key=lambda hit: hit["score"], reverse=True)[:top_k]


def inspect_result(result):
    reasons = list(dict.fromkeys(w for w in result["warnings"] if any(key in w for key in (
        "ocr_", "pdf_", "unresolved", "inferred", "unverified", "missing"))))
    blocks = result.get("parsed_document", {}).get("blocks", [])
    layout_review = any("reading_order" in warning or "pdf_" in warning for warning in result["warnings"])
    context_review = any(b.get("metadata", {}).get("header_context_truncated") for b in blocks)
    checks = {
        "source_fidelity": {"status": "unverified", "reason": "No independent whole-source oracle"},
        "reading_order": {"status": "needs_review" if layout_review else "unverified"},
        "table_context": {"status": "needs_review" if context_review else "unverified"},
        "resources": {"status": "needs_review" if result["deferred_blocks"] else "unverified"},
    }
    return {"chunks": len(result["chunks"]),
            "max_tokens": max((c["token_count"] for c in result["chunks"]), default=0),
            "deferred_blocks": len(result["deferred_blocks"]),
            "warnings": result["warnings"],
            "processing_completed": True,
            "block_coverage": result["completeness"],
            "extraction_quality": "needs_review" if reasons or result["deferred_blocks"] or context_review else "unverified",
            "review_reasons": reasons,
            "table_value_coverage": "validated_by_fusion_cells_v1",
            "parser_quality": {"schema_version": 1, "checks": checks},
            "index_readiness": {"ready": False, "reason": "Source accuracy requires independent verification"},
            "scope": "No automatic claim of source accuracy; lexical search only"}
