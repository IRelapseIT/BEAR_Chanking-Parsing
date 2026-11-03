import hashlib
import importlib.metadata
import json
import time
from dataclasses import asdict, dataclass
from typing import Callable, Iterator
from uuid import UUID, uuid4, uuid5

from chonkie import RecursiveChunker, SemanticChunker
from razdel import sentenize

from .embeddings import InfinityChonkieEmbeddings
from .errors import EmbeddingError, ValidationError
from .models import (
    Chunk,
    ContentBlock,
    EmbeddingProfile,
    ParsedDocument,
    ProcessingProfile,
    ProcessingResult,
    SourceSpan,
)
from .ports import EmbeddingProvider, ExternalContentProcessor, TokenCounter


@dataclass
class Region:
    text: str
    # Ranges in the region that correspond to original blocks; separators are synthetic.
    ranges: list[tuple[int, int, object]]
    section_path: list[str]
    structured: bool
    role: str


@dataclass(frozen=True)
class ChunkDecision:
    strategy: str
    reason: str | None = None


class AdaptiveChunkingStrategy:
    """Choose a branch from usable boundaries, not merely an inherited heading path."""

    @staticmethod
    def choose(region: Region, profile: ProcessingProfile, fits: Callable[[str], bool],
               token_count: Callable[[str], int]) -> ChunkDecision:
        if profile.mode == "fixed":
            return ChunkDecision("fixed")
        if profile.mode == "token":
            return ChunkDecision("recursive")
        if profile.mode == "adaptive" and fits(region.text):
            return ChunkDecision("whole")
        if profile.mode == "adaptive" and region.structured:
            return ChunkDecision("structural")
        semantic = profile.mode == "semantic" or (
            profile.mode == "adaptive" and region.role not in {"code", "list_item"}
            and len(list(sentenize(region.text))) > profile.similarity_window
        )
        if semantic and token_count(region.text) <= profile.semantic_region_tokens:
            return ChunkDecision("semantic")
        if semantic:
            return ChunkDecision("recursive", "semantic_budget_exceeded")
        return ChunkDecision("recursive")


class ContentProfiler:
    def regions(self, document: ParsedDocument) -> Iterator[Region | ContentBlock]:
        pending = []

        def assemble():
            parts, ranges, position = [], [], 0
            for block in pending:
                if parts:
                    parts.append("\n\n")
                    position += 2
                parts.append(block.text)
                ranges.append((position, position + len(block.text), block))
                position += len(block.text)
            first = pending[0]
            substantive = [block for block in pending if block.role != "heading"]
            structured = any(block.role == "list_item" for block in pending) or (
                bool(first.section_path) and len(substantive) >= 2
            )
            return Region("".join(parts), ranges, first.section_path,
                          structured, first.role)

        for block in document.blocks:
            if block.type != "TEXT":
                if pending:
                    yield assemble()
                    pending.clear()
                yield block
                continue
            if pending and (block.section_path != pending[0].section_path
                            or block.role == "code" or pending[0].role == "code"):
                yield assemble()
                pending.clear()
            pending.append(block)
        if pending:
            yield assemble()


class SafeTextSplitter:
    def __init__(self, fits: Callable[[str], bool]):
        self.fits = fits

    def split(self, text: str) -> list[tuple[int, int]]:
        output, offset = [], 0
        while offset < len(text):
            remaining = text[offset:]
            if self.fits(remaining):
                output.append((offset, len(text)))
                break
            # A bounded search is used as a size heuristic; the chosen slice is always checked exactly.
            low, high, best = 1, len(remaining), 0
            while low <= high:
                middle = (low + high) // 2
                if self.fits(remaining[:middle]):
                    best = middle
                    low = middle + 1
                else:
                    high = middle - 1
            if best == 0:
                raise ValidationError("Даже один символ не помещается вместе с контекстом")
            candidate = remaining[:best]
            boundaries = [match.end() for match in __import__("re").finditer(r"\n\n|[.!?]\s+|\s+", candidate)]
            preferred = [point for point in boundaries if point >= best // 2]
            cut = preferred[-1] if preferred else best
            while cut > 0 and not self.fits(remaining[:cut]):
                cut -= 1
            if cut == 0:
                raise ValidationError("Разбиение не уменьшает проблемный фрагмент")
            output.append((offset, offset + cut))
            offset += cut
        return output


class SourceMapper:
    @staticmethod
    def map(region: Region, start: int, end: int) -> list[SourceSpan]:
        return [SourceSpan(block.id, max(start, left) - left, min(end, right) - left)
                for left, right, block in region.ranges if max(start, left) < min(end, right)]

    @staticmethod
    def align(original: str, pieces: list) -> list[tuple[int, int]]:
        if not pieces or "".join(piece.text for piece in pieces) != original:
            raise ValidationError("Chonkie изменил текст или нарушил покрытие")
        position, spans = 0, []
        for piece in pieces:
            if not piece.text:
                raise ValidationError("Chonkie вернул пустой чанк")
            spans.append((position, position + len(piece.text)))
            position += len(piece.text)
        return spans


class StructuralChunker:
    @staticmethod
    def split(region: Region, fits: Callable[[str], bool]) -> list[tuple[int, int]]:
        candidates = []
        begin, finish = 0, 0
        for left, right, _ in region.ranges:
            if finish > begin and not fits(region.text[begin:right]):
                candidates.append((begin, left))
                begin = left
            finish = right
        if finish > begin:
            candidates.append((begin, len(region.text)))
        return candidates


class ChonkieRecursiveAdapter:
    @staticmethod
    def split(text: str, tokenizer: object, budget: int) -> list[tuple[int, int]]:
        chunker = RecursiveChunker(tokenizer=tokenizer, chunk_size=budget,
                                   min_characters_per_chunk=1)
        return SourceMapper.align(text, chunker.chunk(text))


class PreservingSemanticChunker(SemanticChunker):
    # Private hook is intentionally confined to the pinned Chonkie 1.7.0 adapter.
    def _split_sentences(self, text):
        sentences = list(sentenize(text))
        if not sentences:
            return [text] if text else []
        ends = [sentence.start for sentence in sentences[1:]] + [len(text)]
        starts = [0] + [sentence.start for sentence in sentences[1:]]
        return [text[start:end] for start, end in zip(starts, ends, strict=True)]


class ChonkieSemanticAdapter:
    @staticmethod
    def split(text: str, counter: TokenCounter, provider: EmbeddingProvider,
              embedding_profile: EmbeddingProfile, profile: ProcessingProfile,
              budget: int) -> list[tuple[int, int]]:
        embeddings = InfinityChonkieEmbeddings(provider, counter, embedding_profile)
        chunker = PreservingSemanticChunker(
            embedding_model=embeddings, chunk_size=budget,
            threshold=profile.threshold, similarity_window=profile.similarity_window,
            skip_window=0, min_characters_per_sentence=1,
        )
        return SourceMapper.align(text, chunker.chunk(text))


class ChunkContextBuilder:
    @staticmethod
    def embedding_text(text: str, section_path: list[str], counter=None) -> str:
        heading = " → ".join(section_path)
        # Keep full headings in metadata, limiting only their repeated embedding context.
        if counter:
            while heading and counter.count(heading) > 64:
                heading = heading[:max(0, len(heading) // 2)]
        return heading + "\n\n" + text if heading else text


class ChunkValidator:
    def __init__(self, counter, maximum):
        self.counter = counter
        self.maximum = maximum

    def validate(self, result: ProcessingResult) -> None:
        blocks = {block.id: block for block in result.parsed_document.blocks}
        if len(blocks) != len(result.parsed_document.blocks):
            raise ValidationError("Duplicate block IDs")
        covered = {block_id: [] for block_id in blocks}
        external_covered = set()
        seen = set()
        if result.context != result.parsed_document.context or result.context.document_version < 1:
            raise ValidationError("Некорректный контекст документа")
        if any(block.order != i for i, block in enumerate(result.parsed_document.blocks)):
            raise ValidationError("Неверный порядок ContentBlock")
        last_source = (-1, -1)
        for sequence, chunk in enumerate(result.chunks):
            if chunk.sequence != sequence or chunk.id in seen or not chunk.text.strip():
                raise ValidationError("Неверный порядок, ID или пустой чанк")
            seen.add(chunk.id)
            if chunk.id != str(uuid5(UUID(result.generation_id), str(sequence))):
                raise ValidationError("Point ID не соответствует generation и sequence")
            count = self.counter.full_count(chunk.embedding_text)
            if chunk.token_count != count or count > self.maximum:
                raise ValidationError("Неверный token_count или превышение бюджета")
            if not chunk.source_spans:
                raise ValidationError("У чанка нет источников")
            text_cursor = 0
            for span in chunk.source_spans:
                block = blocks.get(span.block_id)
                if block is None:
                    raise ValidationError("Неизвестный SourceSpan")
                if chunk.strategy == "external" and block.type != "TEXT":
                    if not 0 <= span.start_char <= span.end_char <= len(block.text):
                        raise ValidationError("Некорректный внешний SourceSpan")
                    external_covered.add(block.id)
                    continue
                if not 0 <= span.start_char < span.end_char <= len(block.text):
                    raise ValidationError("Некорректный SourceSpan")
                if (block.order, span.start_char) < last_source:
                    raise ValidationError("Нарушен порядок источников")
                last_source = (block.order, span.end_char)
                source_text = block.text[span.start_char:span.end_char]
                found = chunk.text.find(source_text, text_cursor)
                if found < 0 or chunk.text[text_cursor:found].strip():
                    raise ValidationError("Исходный текст не соответствует чанку")
                text_cursor = found + len(source_text)
                covered[block.id].append((span.start_char, span.end_char))
            if chunk.strategy != "external" and chunk.text[text_cursor:].strip():
                raise ValidationError("Чанк содержит текст без источника")
            if chunk.embedding_text != ChunkContextBuilder.embedding_text(
                    chunk.text, chunk.section_path, self.counter):
                raise ValidationError("Embedding text не соответствует тексту и контексту")
        deferred = {block.id for block in result.deferred_blocks}
        if not deferred <= blocks.keys() or any(blocks[key].type == "TEXT" for key in deferred):
            raise ValidationError("Некорректные deferred blocks")
        for block in blocks.values():
            if block.id in deferred:
                continue
            if block.type == "TEXT":
                position = 0
                for start, end in sorted(covered[block.id]):
                    if start != position:
                        raise ValidationError(f"Пропуск/дублирование текста блока {block.id}")
                    position = end
                if position != len(block.text):
                    raise ValidationError(f"Неполное покрытие блока {block.id}")
            elif block.id not in external_covered:
                raise ValidationError("Внешний блок не обработан и не помечен deferred")
        expected = "partial" if deferred else "complete"
        if result.completeness != expected:
            raise ValidationError("Completeness не соответствует deferred blocks")


class ChunkingService:
    def __init__(self, counter: TokenCounter, provider: EmbeddingProvider,
                 embedding_profile: EmbeddingProfile,
                 external_processors: dict[str, ExternalContentProcessor] | None = None):
        self.counter = counter
        self.provider = provider
        self.embedding_profile = embedding_profile
        self.external_processors = external_processors or {}

    def profile_id(self, profile, parser_profile):
        config = {"policy": "adaptive-v1", "profile": asdict(profile), "parser": parser_profile,
                  "embedding": asdict(self.embedding_profile), "chonkie": importlib.metadata.version("chonkie"),
                  "counter": type(self.counter).__name__}
        config["dependencies"] = {name: importlib.metadata.version(name) for name in (
            "bear-text-beta", "markdown-it-py", "beautifulsoup4", "lxml", "razdel",
        )}
        return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:24]

    def process(self, document: ParsedDocument, processing_profile: ProcessingProfile | None = None,
                generation_id: str | None = None) -> ProcessingResult:
        started = time.perf_counter()
        initial_metrics = dict(getattr(self.provider, "metrics", {}))
        profile = processing_profile or ProcessingProfile()
        generation_id = str(UUID(generation_id)) if generation_id else str(uuid4())
        maximum = min(profile.max_tokens, self.embedding_profile.max_input_tokens)
        chunks, deferred = [], []
        warnings = list(document.warnings)
        budget = min(profile.target_tokens, maximum)

        def append_piece(region, start, end, strategy, reason=None, attachments=None, metadata=None):
            text = region.text[start:end]
            if not text.strip():
                # Attach whitespace to the preceding piece, retaining its source ranges.
                if chunks and chunks[-1].section_path == region.section_path:
                    chunks[-1].text += text
                    chunks[-1].source_spans.extend(SourceMapper.map(region, start, end))
                    chunks[-1].embedding_text = ChunkContextBuilder.embedding_text(
                        chunks[-1].text, chunks[-1].section_path, self.counter)
                    chunks[-1].token_count = self.counter.full_count(chunks[-1].embedding_text)
                    return
                raise ValidationError("Не удалось сопоставить отдельный пробельный фрагмент")
            embedding_text = ChunkContextBuilder.embedding_text(text, region.section_path, self.counter)
            sequence = len(chunks)
            attachment_ids = attachments or sorted({attachment for _, _, block in region.ranges
                                                   for attachment in block.attachment_ids})
            chunks.append(Chunk(str(uuid5(UUID(generation_id), str(sequence))), sequence, text,
                                embedding_text, self.counter.full_count(embedding_text),
                                SourceMapper.map(region, start, end), list(region.section_path),
                                attachment_ids, strategy, reason, metadata or {}))

        def process_text(region):
            def fits(text, limit=budget):
                return self.counter.full_count(ChunkContextBuilder.embedding_text(
                    text, region.section_path, self.counter)) <= limit

            decision = AdaptiveChunkingStrategy.choose(region, profile, fits, self.counter.count)
            if decision.strategy == "whole":
                append_piece(region, 0, len(region.text), "whole")
                return
            if decision.strategy == "fixed":
                candidates = [(i, min(i + profile.fixed_characters, len(region.text)))
                              for i in range(0, len(region.text), profile.fixed_characters)]
                strategy, reason = "fixed", None
            elif decision.strategy == "structural":
                candidates = StructuralChunker.split(region, fits)
                strategy, reason = "structural", None
            else:
                semantic = decision.strategy == "semantic"
                strategy, reason = decision.strategy, decision.reason
                try:
                    if semantic:
                        candidates = ChonkieSemanticAdapter.split(
                            region.text, self.counter, self.provider, self.embedding_profile, profile, budget)
                    else:
                        candidates = ChonkieRecursiveAdapter.split(
                            region.text, self.counter.tokenizer, budget)
                except (EmbeddingError, ValidationError, ValueError, RuntimeError) as exc:
                    reason = f"{strategy}_failure:{type(exc).__name__}"
                    if semantic:
                        strategy = "recursive"
                        try:
                            candidates = ChonkieRecursiveAdapter.split(
                                region.text, self.counter.tokenizer, budget)
                        except (ValidationError, ValueError, RuntimeError):
                            candidates = [(0, len(region.text))]
                            reason += ";source_preserving_repair"
                    else:
                        candidates = [(0, len(region.text))]
                        reason += ";source_preserving_repair"
            if reason:
                warnings.append(reason)
            for start, end in candidates:
                part = region.text[start:end]
                if fits(part):
                    append_piece(region, start, end, strategy, reason)
                else:
                    repairs = [(0, len(part))]
                    if strategy == "structural":
                        try:
                            repairs = ChonkieRecursiveAdapter.split(part, self.counter.tokenizer, budget)
                        except (ValidationError, ValueError, RuntimeError):
                            pass
                    for repair_start, repair_end in repairs:
                        safe = SafeTextSplitter(fits)
                        for left, right in safe.split(part[repair_start:repair_end]):
                            append_piece(region, start + repair_start + left,
                                         start + repair_start + right, strategy,
                                         reason or "token_budget_repair")

        for item in ContentProfiler().regions(document):
            if isinstance(item, Region):
                process_text(item)
            else:
                processor = self.external_processors.get(item.type)
                if processor is None:
                    deferred.append(item)
                    warnings.append(f"external_processor_missing:{item.type}:{item.id}")
                    continue
                fragments = processor.process(item, document.context)
                if not fragments:
                    deferred.append(item)
                    warnings.append(f"external_processor_empty:{item.id}")
                    continue
                for fragment in fragments:
                    region = Region(fragment.text, [], item.section_path, True, "external")
                    # External processors own source interpretation and prepared-fragment boundaries.
                    base = len(chunks)
                    section = list(item.section_path)
                    def fits(text, section=section):
                        return self.counter.full_count(ChunkContextBuilder.embedding_text(
                            text, section, self.counter)) <= budget
                    for start, end in SafeTextSplitter(fits).split(fragment.text):
                        append_piece(region, start, end, "external", attachments=fragment.attachment_ids,
                                     metadata=fragment.metadata)
                        chunks[-1].source_spans = fragment.source_spans
                    if len(chunks) == base:
                        raise ValidationError("External fragment is empty")
        result = ProcessingResult(
            document.context, self.profile_id(profile, document.parser_profile), generation_id,
            document, chunks, deferred, list(dict.fromkeys(warnings)), "partial" if deferred else "complete",
            metrics={"processing_seconds": time.perf_counter() - started,
                     "chunk_count": len(chunks), "embedding_profile_id": self.embedding_profile.identifier(),
                     "embedding_profile": asdict(self.embedding_profile),
                     "processing_profile": asdict(profile), "token_counter": type(self.counter).__name__,
                     "dependency_versions": {name: importlib.metadata.version(name) for name in (
                         "bear-text-beta", "chonkie", "chonkie-core", "markdown-it-py",
                         "beautifulsoup4", "lxml", "razdel", "transformers",
                     )},
                     "inference": {key: value - initial_metrics.get(key, 0)
                                   for key, value in getattr(self.provider, "metrics", {}).items()}},
        )
        ChunkValidator(self.counter, maximum).validate(result)
        return result
