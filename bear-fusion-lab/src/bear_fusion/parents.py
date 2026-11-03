"""Stable section graph and bounded local Parent-Child expansion."""
import json
from uuid import NAMESPACE_URL, uuid5


def attach_parents(result):
    context = result.context
    identity = json.dumps([context.tenant_id, context.space_id, context.document_id,
                           context.document_version, context.checksum, result.processing_profile_id])
    identify = lambda anchor: str(uuid5(NAMESPACE_URL, identity + "/" + anchor))
    root = identify("root")
    parents = {root: {"id": root, "parent_id": None, "title": context.filename,
                      "section_path": [], "block_ids": []}}
    stack, owner = [], {}
    for block in result.parsed_document.blocks:
        path = tuple(block.section_path)
        if block.role == "heading":
            while stack and (len(stack[-1][0]) >= len(path) or path[:len(stack[-1][0])] != stack[-1][0]):
                stack.pop()
            parent_id = identify("heading/" + block.id)
            parents[parent_id] = {"id": parent_id, "parent_id": stack[-1][1] if stack else root,
                                  "title": block.text, "section_path": list(path), "block_ids": []}
            stack.append((path, parent_id))
        else:
            while stack and path[:len(stack[-1][0])] != stack[-1][0]:
                stack.pop()
            parent_id = stack[-1][1] if stack else root
        owner[block.id] = parent_id
        parents[parent_id]["block_ids"].append(block.id)
    blocks = {block.id: block for block in result.parsed_document.blocks}
    orders = {}
    code_parts = {}
    for chunk in result.chunks:
        parent_ids = list(dict.fromkeys(owner[span.block_id] for span in chunk.source_spans))
        primary = parent_ids[0]
        chunk.metadata.update(parent_id=primary, parent_ids=parent_ids,
                              chunk_order=orders.get(primary, 0), breadcrumb=" → ".join(chunk.section_path))
        orders[primary] = orders.get(primary, 0) + 1
        codes = [blocks[span.block_id] for span in chunk.source_spans if blocks[span.block_id].role == "code"]
        if codes:
            chunk.metadata["code_languages"] = list(dict.fromkeys(b.metadata.get("language", "") for b in codes))
            chunk.metadata["code_block_ids"] = list(dict.fromkeys(b.id for b in codes))
            for block in codes:
                code_parts.setdefault(block.id, []).append(chunk)
    for block_id, chunks in code_parts.items():
        for index, chunk in enumerate(chunks):
            chunk.metadata.setdefault("code_parts", {})[block_id] = {"part": index + 1, "total": len(chunks)}
    return list(parents.values())


def expand_context(envelope, chunk_id, counter, max_tokens=480):
    if not 1 <= max_tokens <= 4096:
        raise ValueError("Context budget must be 1..4096 tokens")
    result = envelope["result"]
    chunks = result["chunks"]
    seed = next((chunk for chunk in chunks if chunk["id"] == chunk_id), None)
    if seed is None:
        raise ValueError("Chunk not found in this document generation")
    if counter.count(seed["text"]) > max_tokens:
        raise ValueError("Matched chunk exceeds context budget; increase it")
    parents = set(seed["metadata"].get("parent_ids", []))
    candidates = [c for c in chunks if parents.intersection(c["metadata"].get("parent_ids", []))]
    chosen = [seed]
    for candidate in sorted(candidates, key=lambda c: abs(c["sequence"] - seed["sequence"])):
        if candidate["id"] == seed["id"]:
            continue
        trial = sorted([*chosen, candidate], key=lambda c: c["sequence"])
        if counter.count("\n\n".join(c["text"] for c in trial)) <= max_tokens:
            chosen = trial
    chosen.sort(key=lambda c: c["sequence"])
    text = "\n\n".join(c["text"] for c in chosen)
    return {"text": text, "token_count": counter.count(text), "max_tokens": max_tokens,
            "parent_ids": sorted(parents), "chunk_ids": [c["id"] for c in chosen],
            "truncated": len(chosen) < len(candidates),
            "context": result["context"], "generation_id": result["generation_id"],
            "sources": [s for c in chosen for s in c["source_spans"]],
            "code_context": [c["metadata"] for c in chosen if c["metadata"].get("code_languages")]}
