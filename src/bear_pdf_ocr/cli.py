"""Read a local PDF and emit the BEAR parsed-document JSON."""
import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .limits import FileLimits
from .models import DocumentContext
from .pdf_adapter import parse_pdf


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("-o", "--output", type=Path, help="New JSON file. Default: stdout.")
    args = parser.parse_args(argv)
    limits = FileLimits()
    try:
        with args.pdf.open("rb") as source:
            if source.read(5) != b"%PDF-":
                raise ValueError("Input must be a PDF")
            if args.pdf.stat().st_size > limits.max_file_bytes:
                raise ValueError("PDF file size exceeds the limit")
            source.seek(0)
            checksum = hashlib.file_digest(source, "sha256").hexdigest()
        context = DocumentContext("local", "pdf-ocr", checksum, 1, args.pdf.name, "application/pdf", checksum)
        document = parse_pdf(args.pdf, context, limits)
        if len(document.blocks) > limits.max_blocks or sum(len(b.text) for b in document.blocks) > limits.max_text_chars:
            raise ValueError("Parsed document exceeds the output limits")
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(asdict(document), stream, ensure_ascii=False, indent=2)
                stream.write("\n")
        else:
            print(json.dumps(asdict(document), ensure_ascii=True, indent=2))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
