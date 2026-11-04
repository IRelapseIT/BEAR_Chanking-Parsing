"""Map isolated Dedoc objects to the versioned parser wire contract without importing Dedoc."""

import base64
import json
import mimetypes
from pathlib import Path

from .file_validation import FileLimits


def dedoc_to_contract(document, attachment_dir: Path, limits: FileLimits) -> dict:
    if document.content is None:
        raise ValueError("Dedoc did not return document content")
    tables = {table.metadata.uid: table for table in document.content.tables}
    attachments = []
    for attachment in document.attachments:
        metadata = attachment.metadata
        name = Path(str(getattr(metadata, "temporary_file_name", ""))).name
        candidate = (attachment_dir / name).resolve()
        if not name or not candidate.is_relative_to(attachment_dir.resolve()) or not candidate.is_file():
            raise ValueError("Dedoc attachment is missing")
        if candidate.stat().st_size > limits.max_attachment_bytes:
            raise ValueError("Attachment exceeds limit")
        data = candidate.read_bytes()
        if len(data) > limits.max_attachment_bytes:
            raise ValueError("Attachment exceeds limit")
        attachments.append({"uid": str(metadata.uid), "name": str(metadata.file_name),
                            "mime_type": mimetypes.guess_type(metadata.file_name)[0] or "application/octet-stream",
                            "data_b64": base64.b64encode(data).decode("ascii")})
    if len(attachments) > limits.max_attachments or sum(len(a["data_b64"]) * 3 // 4 for a in attachments) > (
            limits.max_total_attachment_bytes):
        raise ValueError("Too many attachments")
    blocks = []
    used_tables = set()

    def table_source(table):
        schema = table.to_api_schema()
        raw = schema.model_dump(mode="json") if hasattr(schema, "model_dump") else schema.dict()
        serialized = json.dumps(raw, ensure_ascii=False)
        if len(serialized.encode("utf-8")) > limits.max_table_inline_bytes:
            raise ValueError("Table exceeds inline limit; streaming table path required")
        return serialized

    def add(node):
        if node.node_id == "0" and not node.text.strip():
            pass
        else:
            level = getattr(node.metadata, "hierarchy_level", None)
            line_type = str(getattr(level, "line_type", "raw_text"))
            role = "heading" if line_type == "header" else (
                "list_item" if line_type in {"list_item", "bullet_list_item"} else "paragraph")
            page_id = getattr(node.metadata, "page_id", None)
            page = page_id + 1 if type(page_id) is int else None
            if page is not None and page > limits.max_pages:
                raise ValueError("Page limit exceeded")
            heading_level = getattr(level, "level_2", None) if role == "heading" else None
            if not isinstance(heading_level, int) or not 1 <= heading_level <= 6:
                heading_level = min(node.node_id.count("."), 6) if role == "heading" else None
            attachment_uids = []
            table_uids = []
            for annotation in node.annotations:
                if annotation.name == "table":
                    table_uids.append(str(annotation.value))
                elif annotation.name == "attachment":
                    attachment_uids.append(str(annotation.value))
            if node.text.strip():
                blocks.append({"type": "TEXT", "role": role, "text": node.text,
                               "node_id": node.node_id, "line_id": node.metadata.line_id,
                               "page": page, "heading_level": heading_level,
                               "attachment_uids": [], "metadata": {"dedoc_line_type": line_type}})
            for uid in table_uids:
                table = tables.get(uid)
                if table is None:
                    raise ValueError("Table annotation refers to missing table")
                used_tables.add(uid)
                blocks.append({"type": "TABLE", "role": "table", "text": "", "node_id": node.node_id,
                               "line_id": node.metadata.line_id, "page": page,
                               "metadata": {"raw_source": table_source(table), "dedoc_uid": uid}})
            for uid in attachment_uids:
                blocks.append({"type": "IMAGE", "role": "image", "text": "", "node_id": node.node_id,
                               "line_id": node.metadata.line_id, "page": page,
                               "attachment_uids": [uid], "metadata": {"dedoc_uid": uid}})
    stack = [document.content.structure]
    visited = 0
    while stack:
        node = stack.pop()
        visited += 1
        if visited > limits.max_blocks:
            raise ValueError("Dedoc tree exceeds node limit")
        add(node)
        stack.extend(reversed(node.subparagraphs))
    warnings = list(document.content.warnings)
    for uid, table in tables.items():
        if uid not in used_tables:
            page_id = table.metadata.page_id
            page = page_id + 1 if type(page_id) is int else None
            if page is not None and page > limits.max_pages:
                raise ValueError("Page limit exceeded")
            blocks.append({"type": "TABLE", "role": "table", "text": "", "node_id": f"table:{uid}",
                           "line_id": None, "page": page,
                           "metadata": {"raw_source": table_source(table), "dedoc_uid": uid}})
            warnings.append(f"table_position_unknown:{uid}")
    if len(blocks) > limits.max_blocks or sum(len(block["text"]) for block in blocks) > limits.max_text_chars:
        raise ValueError("Dedoc output exceeds block or character limit")
    return {"schema_version": 1, "blocks": blocks, "attachments": attachments, "warnings": warnings}
