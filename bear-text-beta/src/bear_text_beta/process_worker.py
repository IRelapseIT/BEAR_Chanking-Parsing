"""Durable beta processor for large uploaded documents."""

from __future__ import annotations

import time
from threading import Event, Thread
from uuid import uuid4

from .errors import ParseError, StorageError, ValidationError
from .file_validation import StoredObject
from .models import DocumentContext, ProcessingProfile
from .runtime import default_services


def run_once(services, *, lease_seconds=600, heartbeat_interval=None) -> bool:
    interval = heartbeat_interval if heartbeat_interval is not None else lease_seconds / 3
    if not 0 < interval < lease_seconds:
        raise ValueError("heartbeat_interval must be positive and shorter than the lease")
    job = services.manifest.claim_next_processing(lease_seconds, owner_id=str(uuid4()))
    if job is None:
        return False
    token = job["claim_token"]
    stop = Event()

    def renew():
        while not stop.wait(interval):
            try:
                services.manifest.heartbeat_processing(job["id"], token, lease_seconds)
            except Exception:
                # The publication transaction independently checks ownership;
                # a failed heartbeat can never authorize stale publication.
                return

    heartbeat = Thread(target=renew, name=f"bear-lease-{job['id']}", daemon=True)
    heartbeat.start()
    try:
        services.process_stored(StoredObject(**job["stored"]), DocumentContext(**job["context"]),
                                ProcessingProfile(**job["profile"]), job["id"],
                                processing_claim=(job["id"], token))
    except Exception as exc:
        try:
            services.manifest.fail_processing(
                job["id"], exc, not isinstance(exc, (ParseError, ValidationError)), token)
        except StorageError:
            pass  # An expired/reassigned owner must not alter the new claim.
    else:
        try:
            services.manifest.complete_processing(job["id"], token)
        except StorageError:
            pass
    finally:
        stop.set()
        heartbeat.join()
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
