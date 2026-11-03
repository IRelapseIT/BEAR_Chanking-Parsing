from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from bear_text_beta.file_validation import FileLimits
from bear_text_beta.models import (
    ContentBlock,
    DocumentContext,
    ParsedDocument,
    SourceLocator,
)
from bear_text_beta.parsing import DocumentNormalizer

UPSTREAM_COMMIT = "a48d88d21ba6ece0fc37eb1f91f18905705dec7f"


def map_tree(payload: dict, context: DocumentContext, limits: FileLimits | None = None) -> ParsedDocument:
    """Map the actual parser tree, not its Markdown rendering; never invent page locations."""
    limits = limits or FileLimits()
    root = payload.get("tree")
    if not isinstance(root, dict):
        raise TypeError("Parser returned no full tree")
    blocks = []
    total = 0
    stack = [(root, [], "tree", 0)]
    visited = 0
    roles = {"header": "heading", "paragraph": "paragraph", "raw_text": "paragraph",
             "list_item": "list_item", "table": "table", "document": "paragraph"}
    while stack:
        node, trail, address, depth = stack.pop()
        visited += 1
        if depth > 128 or visited > limits.max_blocks:
            raise ValueError("Parser tree exceeds depth/node limit")
        kind = node.get("type")
        if kind not in roles:
            raise ValueError(f"Unsupported parser node: {kind!r}")
        text = node.get("text", "")
        children = node.get("children", [])
        if not isinstance(text, str) or not isinstance(children, list):
            raise TypeError("Invalid parser node fields")
        total += len(text)
        if total > limits.max_text_chars:
            raise ValueError("Parser text exceeds limit")
        child_trail = [*trail, text] if kind == "header" and text.strip() else trail
        if text:
            block_id = hashlib.sha256(address.encode()).hexdigest()[:24]
            blocks.append(ContentBlock(
                id=block_id, order=len(blocks), type="TABLE" if kind == "table" else "TEXT",
                role=roles[kind], text=text,
                section_path=child_trail if kind == "header" else trail,
                source=SourceLocator(format="parser-tree", element_path=address),
                metadata={"upstream_commit": UPSTREAM_COMMIT, "node_type": kind,
                          "source_precision": "parsed_tree_only"},
            ))
        for index in range(len(children) - 1, -1, -1):
            if not isinstance(children[index], dict):
                raise TypeError("Invalid child node")
            stack.append((children[index], child_trail, f"{address}/children/{index}", depth + 1))
    document = ParsedDocument(context=context, blocks=blocks,
                              parser_profile=f"upstream-{UPSTREAM_COMMIT}:tree-v1",
                              warnings=["source_fidelity_unverified", "attachments_not_extracted_by_upstream"])
    return DocumentNormalizer().normalize(document, limits)


class ParserProcess:
    """Isolated Python environment, bounded execution; not an OS sandbox."""

    def __init__(self, repository: Path, python: Path, timeout: float = 120,
                 max_result_bytes: int = 64 * 1024 * 1024):
        self.repository = repository.resolve()
        self.python = python.resolve()
        if timeout <= 0 or max_result_bytes <= 0:
            raise ValueError("Limits must be positive")
        self.timeout = timeout
        self.max_result_bytes = max_result_bytes

    def parse(self, source: Path) -> dict:
        worker = Path(__file__).with_name("parser_worker.py")
        with tempfile.TemporaryDirectory(prefix="bear-fusion-") as directory:
            output = Path(directory) / "parsed.json"
            completed = subprocess.run(
                [str(self.python), str(worker), str(self.repository), str(source.resolve()),
                 str(output), UPSTREAM_COMMIT],
                cwd=directory, timeout=self.timeout, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, check=False,
            )
            if not output.exists():
                raise RuntimeError(f"Parser process failed without result: {completed.returncode}")
            if output.stat().st_size > self.max_result_bytes:
                raise ValueError("Parser response exceeds limit")
            body = json.loads(output.read_text(encoding="utf-8"))
            if completed.returncode or "error" in body:
                raise RuntimeError(f"Parser failed: {body.get('error', 'unknown')}")
            return body
