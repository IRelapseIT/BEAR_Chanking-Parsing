import copy
import io
import multiprocessing
import sqlite3
import time
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import pytest

from bear_text_beta.errors import StorageError
from bear_text_beta.file_validation import StoredObject
from bear_text_beta.models import DocumentContext, ProcessingProfile
from bear_text_beta.process_worker import run_once
from bear_text_beta.runtime import Services
from bear_text_beta.storage import SQLiteManifest


def submit(manifest, job_id="job", version=1):
    stored = StoredObject("original", 4, "checksum")
    context = DocumentContext("tenant", "space", "document", version, "a.txt")
    manifest.submit_processing(job_id, stored, context, ProcessingProfile())
    return stored, context


def expire(manifest, job_id="job"):
    with manifest.connection() as db:
        db.execute("UPDATE processing_jobs SET lease_until=0 WHERE id=?", (job_id,))


def test_final_expired_attempt_becomes_terminal_and_next_job_runs(tmp_path):
    manifest = SQLiteManifest(tmp_path / "state.db")
    submit(manifest)
    for attempt in range(1, 4):
        claim = manifest.claim_next_processing()
        assert claim["attempt"] == attempt
        expire(manifest)
    submit(manifest, "next")
    assert manifest.claim_next_processing()["id"] == "next"
    terminal = manifest.processing_status("job")
    assert terminal["status"] == "failed"
    assert terminal["error"] == "lease_exhausted"
    assert terminal["completed_at"] is not None


@pytest.mark.parametrize("operation", ["complete", "fail", "heartbeat"])
def test_old_owner_cannot_mutate_reclaimed_job(tmp_path, operation):
    manifest = SQLiteManifest(tmp_path / "state.db")
    submit(manifest)
    old = manifest.claim_next_processing(owner_id="old")
    expire(manifest)
    current = manifest.claim_next_processing(owner_id="current")
    assert current["claim_token"] != old["claim_token"]
    with pytest.raises(StorageError, match="lease"):
        if operation == "complete":
            manifest.complete_processing("job", old["claim_token"])
        elif operation == "fail":
            manifest.fail_processing("job", "stale error", True, old["claim_token"])
        else:
            manifest.heartbeat_processing("job", old["claim_token"])
    assert manifest.processing_status("job")["status"] == "running"
    manifest.complete_processing("job", current["claim_token"])
    assert manifest.processing_status("job")["status"] == "complete"


def test_missing_token_and_expired_heartbeat_fail_closed(tmp_path):
    manifest = SQLiteManifest(tmp_path / "state.db")
    submit(manifest)
    claim = manifest.claim_next_processing()
    with pytest.raises(StorageError):
        manifest.complete_processing("job")
    expire(manifest)
    with pytest.raises(StorageError):
        manifest.heartbeat_processing("job", claim["claim_token"])


def test_retry_counts_and_terminal_failure(tmp_path):
    manifest = SQLiteManifest(tmp_path / "state.db")
    submit(manifest)
    for attempt in range(1, 4):
        claim = manifest.claim_next_processing()
        manifest.fail_processing("job", "temporary", True, claim["claim_token"])
        assert manifest.processing_status("job")["status"] == ("pending" if attempt < 3 else "failed")
    assert manifest.claim_next_processing() is None


def test_live_heartbeat_keeps_a_long_job_owned(tmp_path):
    manifest = SQLiteManifest(tmp_path / "state.db")
    submit(manifest)
    calls = []

    def process(stored, context, profile, generation_id, *, processing_claim):
        # Extend beyond the original lease using the real heartbeat thread.
        time.sleep(0.9)
        manifest.assert_processing_claim(*processing_claim)
        assert manifest.claim_next_processing() is None
        calls.append(generation_id)

    assert run_once(SimpleNamespace(manifest=manifest, process_stored=process),
                    lease_seconds=0.6, heartbeat_interval=0.05)
    assert calls == ["job"]
    assert manifest.processing_status("job")["status"] == "complete"


def test_stale_publication_is_rejected_before_checkpoint_and_outbox(tmp_path, parsed, chunker):
    manifest = SQLiteManifest(tmp_path / "state.db")
    document = parsed("content", "a.txt")
    stored = StoredObject("original", 7, document.context.checksum)
    manifest.submit_processing("job", stored, document.context, ProcessingProfile())
    old = manifest.claim_next_processing()
    result = chunker.process(document, generation_id="46a4f067-d571-432e-a81b-39261e7c4fa1")
    # A real generation UUID is required, and the job uses that same identity.
    with manifest.connection() as db:
        db.execute("UPDATE processing_jobs SET id=? WHERE id='job'", (result.generation_id,))
    expire(manifest, result.generation_id)
    current = manifest.claim_next_processing()
    with pytest.raises(StorageError):
        manifest.save_processing_result(result.generation_id, old["claim_token"], result)
    with pytest.raises(StorageError, match="lease"):
        manifest.save(result)
    with pytest.raises(StorageError, match="не найден"):
        manifest.load(result.generation_id)
    assert manifest.pending_outbox() == []
    manifest.save_processing_result(result.generation_id, current["claim_token"], result)
    assert manifest.load(result.generation_id).context == result.context
    assert manifest.pending_outbox() == [result.generation_id]


def test_publication_rechecks_lease_after_serialization_and_rolls_back(tmp_path, parsed, chunker, monkeypatch):
    manifest = SQLiteManifest(tmp_path / "state.db")
    document = parsed("content", "a.txt")
    result = chunker.process(document)
    manifest.submit_processing(result.generation_id, StoredObject("key", 7, document.context.checksum),
                               document.context, ProcessingProfile())
    claim = manifest.claim_next_processing()
    original = manifest._save_result

    def expire_during_save(db, output):
        original(db, output)
        db.execute("UPDATE processing_jobs SET lease_until=0 WHERE id=?", (claim["id"],))

    monkeypatch.setattr(manifest, "_save_result", expire_during_save)
    with pytest.raises(StorageError):
        manifest.save_processing_result(claim["id"], claim["claim_token"], result)
    with pytest.raises(StorageError):
        manifest.load(result.generation_id)
    assert manifest.pending_outbox() == []


def test_processing_publication_requires_exact_job_context(tmp_path, parsed, chunker):
    manifest = SQLiteManifest(tmp_path / "state.db")
    document = parsed("content", "a.txt")
    result = chunker.process(document)
    manifest.submit_processing(result.generation_id, StoredObject("key", 7, document.context.checksum),
                               document.context, ProcessingProfile())
    claim = manifest.claim_next_processing()
    bad = copy.deepcopy(result)
    bad.context.tenant_id = "other"
    with pytest.raises(StorageError, match="заявке"):
        manifest.save_processing_result(claim["id"], claim["claim_token"], bad)
    manifest.save_processing_result(claim["id"], claim["claim_token"], result)


def test_services_does_not_publish_if_lease_is_lost_during_processing(tmp_path):
    services = Services(tmp_path / "runtime", tmp_path / "unused", test_counter=True)
    try:
        context = DocumentContext("t", "s", "d", 1, "a.txt")
        stored = services.upload_stream(io.BytesIO(b"content"), context)
        job_id = str(uuid4())
        services.manifest.submit_processing(job_id, stored, context, ProcessingProfile())
        old = services.manifest.claim_next_processing()
        parser = services.parser

        class LosingParser:
            max_file_bytes = parser.max_file_bytes

            def parse(self, data, ctx):
                output = parser.parse(data, ctx)
                expire(services.manifest, job_id)
                services.manifest.claim_next_processing(owner_id="replacement")
                return output

        services.parser = LosingParser()
        try:
            with pytest.raises(StorageError, match="lease"):
                services.process_stored(stored, context, ProcessingProfile(), job_id,
                                        processing_claim=(job_id, old["claim_token"]))
            with pytest.raises(StorageError):
                services.manifest.load(job_id)
        finally:
            services.parser = parser
    finally:
        services.close()


def test_reordered_start_cannot_override_a_newer_desired_version(tmp_path, parsed, chunker):
    manifest = SQLiteManifest(tmp_path / "state.db")
    first = chunker.process(parsed("version one"))
    second_doc = parsed("version two")
    second_doc.context.document_version = 2
    second = chunker.process(second_doc)
    manifest.save(first)
    manifest.save(second)
    manifest.begin_index(second.generation_id)
    with pytest.raises(StorageError, match="устаревшую"):
        manifest.begin_index(first.generation_id)
    manifest.activate(second.generation_id)
    assert manifest.active_generations("tenant-a", "space-a") == [second.generation_id]


def test_newly_accepted_processing_version_fences_old_activation(tmp_path, parsed, chunker):
    manifest = SQLiteManifest(tmp_path / "state.db")
    first = chunker.process(parsed("old"))
    manifest.save(first)
    manifest.begin_index(first.generation_id)
    context = replace(first.context, document_version=2)
    manifest.submit_processing("new-job", StoredObject("key", 1, "new-checksum"), context,
                               ProcessingProfile())
    with pytest.raises(StorageError):
        manifest.activate(first.generation_id)
    assert manifest.active_generations("tenant-a", "space-a") == []


def test_legacy_database_migration_preserves_pending_and_recovers_unfenced_running(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE processing_jobs (id TEXT PRIMARY KEY, object_json TEXT NOT NULL, "
                   "context_json TEXT NOT NULL, profile_json TEXT NOT NULL, status TEXT NOT NULL, "
                   "attempts INTEGER NOT NULL DEFAULT 0, claimed_at REAL, error TEXT)")
        for job_id, attempts in [("recover", 1), ("exhausted", 3)]:
            db.execute("INSERT INTO processing_jobs VALUES (?,?,?,?,?,?,?,?)",
                       (job_id, "{}", "{}", "{}", "running", attempts, 1.0, None))
    manifest = SQLiteManifest(path)
    # Opening it again must be harmless and must not reclaim valid fenced owners.
    assert manifest.processing_status("recover")["status"] == "pending"
    assert manifest.processing_status("exhausted")["status"] == "failed"
    claim = manifest.claim_next_processing()
    SQLiteManifest(path)
    manifest.assert_processing_claim(claim["id"], claim["claim_token"])


def _claim_in_process(path, owner, gate, result):
    manifest = SQLiteManifest(path)
    gate.wait(10)
    claim = manifest.claim_next_processing(owner_id=owner)
    result.put(claim)


def _stale_publication_in_process(path, output, ready, release, result):
    manifest = SQLiteManifest(path)
    claim = manifest.claim_next_processing(owner_id="old-process")
    ready.put(claim)
    if not release.wait(15):
        raise RuntimeError("Parent did not release stale worker")
    try:
        manifest.save_processing_result(claim["id"], claim["claim_token"], output)
    except StorageError:
        result.put("rejected")
    else:
        result.put("published")


def test_reclaimed_owner_in_another_process_cannot_publish(tmp_path, parsed, chunker):
    path = tmp_path / "publication-race.db"
    manifest = SQLiteManifest(path)
    document = parsed("original content", "a.txt")
    output = chunker.process(document)
    manifest.submit_processing(output.generation_id, StoredObject("key", 16, document.context.checksum),
                               document.context, ProcessingProfile())
    ctx = multiprocessing.get_context("spawn")
    ready, result, release = ctx.Queue(), ctx.Queue(), ctx.Event()
    worker = ctx.Process(target=_stale_publication_in_process,
                         args=(path, output, ready, release, result))
    try:
        worker.start()
        old = ready.get(timeout=20)
        expire(manifest, output.generation_id)
        current = manifest.claim_next_processing(owner_id="replacement-process")
        assert current["claim_token"] != old["claim_token"]
        release.set()
        assert result.get(timeout=20) == "rejected"
        worker.join(20)
        assert worker.exitcode == 0
        assert manifest.pending_outbox() == []
        with pytest.raises(StorageError):
            manifest.load(output.generation_id)
        manifest.save_processing_result(current["id"], current["claim_token"], output)
        manifest.complete_processing(current["id"], current["claim_token"])
        assert manifest.processing_status(current["id"])["status"] == "complete"
    finally:
        release.set()
        if worker.is_alive():
            worker.terminate()
        worker.join(5)
        ready.close()
        result.close()


def test_two_processes_cannot_claim_the_same_job(tmp_path):
    path = tmp_path / "race.db"
    manifest = SQLiteManifest(path)
    submit(manifest)
    ctx = multiprocessing.get_context("spawn")
    gate, result = ctx.Event(), ctx.Queue()
    workers = [ctx.Process(target=_claim_in_process, args=(path, str(i), gate, result)) for i in range(2)]
    try:
        for worker in workers:
            worker.start()
        gate.set()
        claims = [result.get(timeout=20) for _ in workers]
        for worker in workers:
            worker.join(20)
            assert worker.exitcode == 0
        assert sum(claim is not None for claim in claims) == 1
        winning = next(claim for claim in claims if claim)
        manifest.complete_processing("job", winning["claim_token"])
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(5)
        result.close()
