from collections import Counter


def boundaries(result):
    return [[(s.block_id, s.start_char, s.end_char) for s in chunk.source_spans] for chunk in result.chunks]


def coverage(result):
    count = sum(s.end_char - s.start_char for chunk in result.chunks for s in chunk.source_spans)
    expected = sum(len(block.text) for block in result.parsed_document.blocks if block.type == "TEXT")
    return {"covered_characters": count, "expected_text_characters": expected,
            "deferred_blocks": len(result.deferred_blocks)}


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
