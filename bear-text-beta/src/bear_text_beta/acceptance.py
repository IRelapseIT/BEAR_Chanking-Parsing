"""Automatic operational acceptance, separate from independent source verification."""
import json


def assess(result):
    reasons = []
    if not result.chunks:
        return {"status": "rejected", "accepted_for_indexing": False, "reasons": ["no_searchable_content"],
                "source_fidelity": "unverified", "policy": "operational-v2"}
    from .resources import optional_badge
    optional = [b for b in result.deferred_blocks if optional_badge(b)]
    required = [b for b in result.deferred_blocks if not optional_badge(b)]
    if required or (result.completeness != "complete" and not optional):
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
        elif warning.startswith("pdf_field_needs_review:"):
            reasons.append("pdf_required_field_needs_review")
        elif warning.startswith("pdf_page_route:"):
            route = json.loads(warning.split(":", 1)[1])
            if route["status"] == "needs_review":
                reasons.append("pdf_page_needs_review")
        elif any(key in warning for key in ("unsupported", "unresolved", "unknown_glyph", "complex_markup",
                                           "include_not", "include_unresolved", "visual_content_not_interpreted")):
            reasons.append(warning)
    reasons = sorted(set(reasons))
    optional_refs = [{"block_id": b.id, "url": b.metadata["resource_url"], "alt": b.text,
                      "values_verified": False} for b in optional]
    review_items = [{"reason": reason, "action": (
        "independent_ocr_verification" if reason == "ocr_requires_review" else
        "image_or_embedded_object_processor" if "embedded" in reason or reason == "unresolved_or_partial_content" else
        "specialized_parser_or_source_review")} for reason in reasons]
    return {"status": "needs_review" if reasons else "accepted", "accepted_for_indexing": not reasons,
            "reasons": reasons, "review_items": review_items, "optional_references": optional_refs, "source_fidelity": "unverified", "policy": "operational-v2",
            "limitations": ["Покрытие извлечённых блоков не доказывает полноту оригинала",
                            "Автоматическая приёмка не является независимой проверкой фактов"] + (
                ["Значения внешних статусных бейджей не загружены; индексируется основной текст"] if optional else [])}
