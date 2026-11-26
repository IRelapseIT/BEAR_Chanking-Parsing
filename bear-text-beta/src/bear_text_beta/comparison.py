from collections import Counter


def boundaries(result):
    return [[(s.block_id, s.start_char, s.end_char) for s in chunk.source_spans] for chunk in result.chunks]


def coverage(result):
    blocks = result.parsed_document.blocks
    text_blocks = {block.id: len(block.text) for block in blocks if block.type == "TEXT"}
    spans = [span for chunk in result.chunks for span in chunk.source_spans]
    spans_by_block = {}
    for span in spans:
        spans_by_block.setdefault(span.block_id, []).append((span.start_char, span.end_char))
    count = 0
    for block_id, length in text_blocks.items():
        end = 0
        for start, stop in sorted(spans_by_block.get(block_id, [])):
            stop = min(length, stop)
            count += max(0, stop - max(end, start))
            end = max(end, stop)
    return {
        "covered_characters": count, "expected_text_characters": sum(text_blocks.values()),
        "blocks_represented": len(spans_by_block), "blocks_total": len(blocks),
        "pages_with_text": sorted({block.source.page for block in blocks if block.source.page and block.text.strip()}),
        "deferred_blocks": len(result.deferred_blocks), "source_fidelity": "unverified",
    }


def compare_results(left, right):
    return {
        "same_document": (left.context.checksum == right.context.checksum
                          and left.context.document_id == right.context.document_id),
        "same_boundaries": boundaries(left) == boundaries(right),
        "left": {"chunks": len(left.chunks), "token_counts": [c.token_count for c in left.chunks],
                 "strategies": dict(Counter(c.strategy for c in left.chunks)),
                 "coverage": coverage(left), "warnings": left.warnings, "metrics": left.metrics},
        "right": {"chunks": len(right.chunks), "token_counts": [c.token_count for c in right.chunks],
                  "strategies": dict(Counter(c.strategy for c in right.chunks)),
                  "coverage": coverage(right), "warnings": right.warnings, "metrics": right.metrics},
    }
