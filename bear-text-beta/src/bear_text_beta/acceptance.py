"""Automatic operational acceptance, separate from independent source verification."""
import json


def assess(result):
    reasons = []
    if not result.chunks:
        return {"status": "rejected", "accepted_for_indexing": False, "reasons": ["no_searchable_content"],
                "source_fidelity": "unverified", "policy": "operational-v1"}
    if result.deferred_blocks or result.completeness != "complete":
        reasons.append("unresolved_or_partial_content")
    for block in result.parsed_document.blocks:
        if block.metadata.get("unresolved"):
            reasons.append("unresolved_block")
        route = block.metadata.get("pdf_route")
        if route and route["status"] == "needs_review":
            reasons.append("pdf_page_needs_review")
        if block.metadata.get("extraction") == "ocr":
            reasons.append("ocr_requires_review")
        if block.metadata.get("header_context_truncated"):
            reasons.append("truncated_table_context")
    for warning in result.parsed_document.warnings:
        if warning.startswith(("ocr_unverified", "ocr_no_text", "pdf_sparse_text_visual")):
            # A successful fallback must not erase the original OCR/visual risk.
            reasons.append("ocr_requires_review")
        elif warning.startswith("pdf_page_route:"):
            route = json.loads(warning.split(":", 1)[1])
            if route["status"] == "needs_review":
                reasons.append("pdf_page_needs_review")
        elif any(key in warning for key in ("unsupported", "unresolved", "unknown_glyph", "complex_markup",
                                           "include_not", "include_unresolved", "visual_content_not_interpreted")):
            reasons.append(warning)
    reasons = sorted(set(reasons))
    return {"status": "needs_review" if reasons else "accepted", "accepted_for_indexing": not reasons,
            "reasons": reasons, "source_fidelity": "unverified", "policy": "operational-v1",
            "limitations": ["Покрытие извлечённых блоков не доказывает полноту оригинала",
                            "Автоматическая приёмка не является независимой проверкой фактов"]}
