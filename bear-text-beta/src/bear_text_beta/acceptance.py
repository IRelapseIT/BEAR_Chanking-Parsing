"""Operational acceptance and structured quality diagnostics."""
import json
import re

from .resources import optional_badge

POLICY = "operational-v3"


def _route(warning):
    code, _, value = warning.partition(":")
    if code not in {"pdf_page_route", "pdf_field_route"}:
        return None
    try:
        data = json.loads(value)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def assess(result):
    diagnostics = []
    document = result.parsed_document
    routes = {}
    for warning in document.warnings:
        route = _route(warning)
        if warning.startswith("pdf_page_route:") and route is not None:
            routes[route.get("page")] = route

    def add(code, message, *, stage="parsing", severity="warning", page=None, block_id=None,
            parser=None, resolution=None, details=None):
        diagnostics.append(dict(code=code, severity=severity, message=message, stage=stage,
                                parser=parser or document.parser_profile, page=page, block_id=block_id,
                                resolved=resolution is not None, resolution=resolution, details=details or {}))

    for warning in dict.fromkeys(document.warnings):
        code = warning.split(":", 1)[0]
        route = _route(warning)
        page_match = re.search(r":page:(\d+)", warning)
        page = int(page_match[1]) if page_match else None
        if route is not None:
            review = route.get("status") in {"needs_review", "blank_in_source"} and route.get("required", True)
            add(code, code, severity="warning" if review else "info", page=route.get("page"),
                parser=route.get("selected_parser") or route.get("engine"), details=route)
            continue
        chosen = routes.get(page, {})
        resolved = (chosen.get("fallback_status") == "selected"
                    and code in chosen.get("primary_reasons", [])
                    and code not in chosen.get("remaining_reasons", []))
        add(code, warning, page=page,
            resolution=chosen.get("selected_parser") if resolved else None)
    for block in document.blocks:
        source = dict(page=block.source.page, block_id=block.id,
                      parser=block.metadata.get("parser") or block.metadata.get("engine"))
        if block.metadata.get("extraction") == "ocr":
            add("ocr_requires_review", "Текст распознан OCR. Точность требует сверки с оригиналом.",
                **source)
        if block.metadata.get("unresolved"):
            add("unresolved_block", "Содержимое области извлечено не полностью.", **source)
        if block.metadata.get("header_context_truncated"):
            add("truncated_table_context", "Контекст заголовков таблицы сокращён.", **source)
    optional_refs = []
    for block in result.deferred_blocks:
        if optional_badge(block):
            optional_refs.append({"block_id": block.id, "url": block.metadata["resource_url"],
                                  "alt": block.text, "values_verified": False})
        else:
            add("unresolved_or_partial_content", "Блок отложен при чанкинге.",
                stage="chunking", page=block.source.page, block_id=block.id)
    for warning in dict.fromkeys(result.warnings):
        if warning not in document.warnings:
            add(warning.split(":", 1)[0], warning, stage="chunking",
                severity="info" if optional_refs and warning.startswith("external_processor_missing:IMAGE:")
                and any(warning.endswith(":" + ref["block_id"]) for ref in optional_refs) else "warning")
    searchable = any(chunk.text.strip() for chunk in result.chunks)
    if not searchable:
        add("no_searchable_content", "Чанкинг не дал пригодного текста.", stage="chunking", severity="error")
    reasons = sorted({item["code"] for item in diagnostics
                      if not item["resolved"] and item["severity"] in {"warning", "error"}})
    outcome = ("failed" if not searchable else "completed_with_warnings" if reasons else "completed")
    return dict(status="accepted" if searchable else "rejected", accepted_for_indexing=searchable,
                outcome=outcome, reasons=reasons, warnings=diagnostics, optional_references=optional_refs,
                source_fidelity="unverified", policy=POLICY,
                limitations=["Покрытие извлечённых блоков не доказывает полноту оригинала",
                             "Автоматическая приёмка не является независимой проверкой фактов"])
