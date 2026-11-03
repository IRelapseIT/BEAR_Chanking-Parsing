import hashlib
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import bear_text_beta
from bear_text_beta.file_validation import FileLimits, FileValidator
from bear_text_beta.models import DocumentContext, ProcessingProfile
from bear_text_beta.parsing import DocumentNormalizer, TXTParser

from .bridge import UPSTREAM_COMMIT, map_tree
from .html_adapter import parse_structured_html
from .markdown_adapter import StructuredMarkdownParser
from .parents import attach_parents
from .preflight import reject_known_table_loss
from .tables import (
    TableProcessor,
    canonical,
    restore_docx_tables,
    validate_table_chunks,
)


class FusionPipeline:
    def __init__(self, parser, chunker, storage, limits: FileLimits | None = None, on_stage=None):
        self.parser, self.chunker, self.storage = parser, chunker, storage
        self.limits = limits or FileLimits()
        self.on_stage = on_stage or (lambda name, data: None)
        if hasattr(chunker, "external_processors"):
            chunker.external_processors.setdefault("TABLE", TableProcessor())

    def process_path(self, path: Path, context: DocumentContext,
                     profile: ProcessingProfile | None = None) -> dict:
        """Return an experimental result; never mark it ready for production indexing."""
        profile = profile or ProcessingProfile()
        if path.name != context.filename:
            raise ValueError("Filename does not match context")
        if context.document_version < 1 or not all((context.tenant_id, context.space_id, context.document_id)):
            raise ValueError("Invalid document context")
        context.mime_type = {".txt": "text/plain", ".md": "text/markdown", ".html": "text/html",
                             ".pdf": "application/pdf",
                             ".htm": "text/html", ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}.get(
                                 path.suffix.lower(), context.mime_type)
        # Physical keys are independent of untrusted tenant/file names.
        key = "originals/" + hashlib.sha256(
            f"{context.tenant_id}/{context.space_id}/{context.document_id}/{context.document_version}".encode()
        ).hexdigest() + "/" + str(uuid4())
        validator = FileValidator(self.limits, allowed_extensions=FileValidator.ALLOWED | {".pdf"})
        with path.open("rb") as stream:
            stored = validator.validate_and_store(stream, context.filename, key, self.storage,
                                                  expected_checksum=context.checksum)
        context.checksum = stored.checksum
        self.on_stage("original", asdict(stored))
        with tempfile.TemporaryDirectory(prefix="bear-fusion-input-") as directory:
            source = Path(directory) / ("input" + path.suffix.lower())
            with self.storage.get_stream(key) as stream, source.open("wb") as target:
                shutil.copyfileobj(stream, target, length=1024 * 1024)
            if source.suffix == ".docx":
                validator.validate_docx(source.read_bytes())
            document = (parse_structured_html(source.read_bytes(), context)
                        if source.suffix in {".html", ".htm"} else None)
            if source.suffix == ".pdf":
                from .pdf_adapter import parse_pdf
                with source.open("rb") as pdf_stream:
                    signature = pdf_stream.read(5)
                if signature != b"%PDF-":
                    raise ValueError("Invalid PDF signature")
                document = parse_pdf(source, context, self.limits)
                parsed = {"runtime": {"route": "pdf-cpu", "ocr": "tesseract-if-no-text"}}
            elif document is not None:
                parsed = {"runtime": {"route": "native-html-structure", "reason": "E05: preserve pre and table structure"}}
            elif source.suffix == ".md":
                document = StructuredMarkdownParser().parse(source.read_bytes(), context)
                document.warnings.append("source_fidelity_unverified")
                parsed = {"runtime": {"route": "native-markdown", "reason": "experiment E02: preserve code and table raw source"}}
            else:
                if source.suffix != ".docx":
                    reject_known_table_loss(source)
                if source.suffix == ".txt":
                    document = TXTParser().parse(source.read_bytes(), context)
                    parsed = {"runtime": {"route": "native-txt", "reason": "exact source text; no Dedoc startup"}}
                else:
                    parsed = self.parser.parse(source)
                    document = map_tree(parsed, context, self.limits)
                if source.suffix == ".docx":
                    restore_docx_tables(source, document)
        document = DocumentNormalizer().normalize(document, self.limits)
        document.parser_profile += ":fusion-cells-v2:parents-v1"
        table_evidence = {}
        for block in document.blocks:
            if block.type == "TABLE":
                cells, _ = canonical(block)
                table_evidence[block.id] = tuple(tuple(vars(cell).items()) for cell in cells)
                if block.metadata.get("header_inference"):
                    document.warnings.append("table_header_inferred")
        self.on_stage("parsed", asdict(document))
        if any(len(block.metadata.get("raw_source", "").encode("utf-8")) > self.limits.max_table_inline_bytes
               for block in document.blocks if block.type == "TABLE"):
            raise ValueError("Deferred table exceeds inline limit; streaming processor required")
        processor = getattr(self.chunker, "external_processors", {}).get("TABLE")
        budget = min(profile.target_tokens, profile.max_tokens)
        if isinstance(processor, TableProcessor):
            processor.counter = self.chunker.counter
            processor.budget = budget
        result = self.chunker.process(document, profile)
        if hasattr(result, "chunks"):
            for block in document.blocks:
                if block.type == "TABLE" and table_evidence[block.id] != tuple(tuple(vars(cell).items()) for cell in canonical(block)[0]):
                    raise ValueError("External processor mutated canonical source cells")
            validate_table_chunks(document, result.chunks, self.chunker.counter, budget)
            parents = attach_parents(result)
        self.on_stage("chunked", asdict(result))
        core_hash = hashlib.sha256()
        core_root = Path(bear_text_beta.__file__).parent
        for module in sorted(core_root.glob("*.py")):
            core_hash.update(module.name.encode())
            core_hash.update(module.read_bytes())
        envelope = {"schema_version": 1, "upstream_commit": UPSTREAM_COMMIT,
                "beta_core_sha256": core_hash.hexdigest(),
                "accepted_for_indexing": False, "source_fidelity": "unverified",
                "original": asdict(stored), "parser_runtime": parsed.get("runtime"),
                "result": asdict(result)}
        if hasattr(result, "chunks"):
            from .quality import inspect_result
            envelope["quality"] = inspect_result(envelope["result"])
            envelope["parents"] = parents
        return envelope
