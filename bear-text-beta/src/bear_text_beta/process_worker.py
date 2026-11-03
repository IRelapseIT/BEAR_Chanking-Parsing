"""Durable beta processor for large uploaded documents."""

from __future__ import annotations

import time

from .errors import ParseError, ValidationError
from .file_validation import StoredObject
from .models import DocumentContext, ProcessingProfile
from .runtime import default_services


def run_once(services) -> bool:
    job = services.manifest.claim_next_processing()
    if job is None:
        return False
    try:
        services.process_stored(StoredObject(**job["stored"]), DocumentContext(**job["context"]),
                                ProcessingProfile(**job["profile"]), job["id"])
    except Exception as exc:
        services.manifest.fail_processing(job["id"], exc, not isinstance(exc, (ParseError, ValidationError)))
    else:
        services.manifest.complete_processing(job["id"])
    return True


def main():
    services = default_services()
    try:
        while True:
            if not run_once(services):
                time.sleep(1)
    finally:
        services.close()


if __name__ == "__main__":
    main()
