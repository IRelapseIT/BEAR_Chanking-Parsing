import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from .errors import StorageError
from .serialization import load_result


class SQLiteManifest:
    """Local job/checkpoint/activation state, replaceable with the team's PostgreSQL adapter."""
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            schema = """
                CREATE TABLE IF NOT EXISTS documents (
                    tenant TEXT, space TEXT, document TEXT,
                    active_generation TEXT, desired_generation TEXT, deleted INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (tenant, space, document)
                );
                CREATE TABLE IF NOT EXISTS versions (
                    tenant TEXT, space TEXT, document TEXT, version INTEGER, checksum TEXT,
                    PRIMARY KEY (tenant, space, document, version)
                );
                CREATE TABLE IF NOT EXISTS generations (
                    id TEXT PRIMARY KEY, tenant TEXT, space TEXT, document TEXT, version INTEGER,
                    profile TEXT, result_json TEXT NOT NULL, digest TEXT NOT NULL,
                    status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, error TEXT
                );
                CREATE TABLE IF NOT EXISTS outbox (
                    job_id TEXT PRIMARY KEY, published INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS processing_jobs (
                    id TEXT PRIMARY KEY, object_json TEXT NOT NULL, context_json TEXT NOT NULL,
                    profile_json TEXT NOT NULL, status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
                    claimed_at REAL, error TEXT
                );
            """
            for statement in schema.split(";"):
                if statement.strip():
                    db.execute(statement)
            # ALTERs run under the same write lock as initialization, including
            # when two workers first open an older manifest concurrently.
            columns = {row["name"] for row in db.execute("PRAGMA table_info(documents)")}
            if "desired_version" not in columns:
                db.execute("ALTER TABLE documents ADD COLUMN desired_version INTEGER NOT NULL DEFAULT 0")
                db.execute("UPDATE documents SET desired_version=COALESCE((SELECT MAX(version) "
                           "FROM generations g WHERE g.tenant=documents.tenant AND g.space=documents.space "
                           "AND g.document=documents.document),0)")
            columns = {row["name"] for row in db.execute("PRAGMA table_info(processing_jobs)")}
            legacy_jobs = "claim_token" not in columns
            for name, sql_type in (("claim_token", "TEXT"), ("owner_id", "TEXT"),
                                   ("lease_until", "REAL"), ("completed_at", "REAL")):
                if name not in columns:
                    db.execute(f"ALTER TABLE processing_jobs ADD COLUMN {name} {sql_type}")
            if legacy_jobs:
                # Pre-fencing workers must be stopped before this migration.
                # They cannot safely retain ownership in the new protocol.
                db.execute("UPDATE processing_jobs SET status=CASE WHEN attempts>=3 THEN 'failed' "
                           "ELSE 'pending' END, error='legacy_lease_recovered', claimed_at=NULL, "
                           "completed_at=CASE WHEN attempts>=3 THEN ? ELSE NULL END WHERE status='running'",
                           (time.time(),))
            db.execute("CREATE INDEX IF NOT EXISTS processing_jobs_claimable "
                       "ON processing_jobs(status,lease_until)")

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def save(self, result):
        with self.connection() as db:
            processing = db.execute("SELECT id FROM processing_jobs WHERE id=?",
                                    (result.generation_id,)).fetchone()
            existing = db.execute("SELECT id FROM generations WHERE id=?", (result.generation_id,)).fetchone()
            if processing is not None and existing is None:
                raise StorageError("Для публикации processing result требуется текущая lease")
            self._save_result(db, result)

    def _save_result(self, db, result):
        serialized = json.dumps(asdict(result), ensure_ascii=False, sort_keys=True, allow_nan=False)
        digest = hashlib.sha256(serialized.encode()).hexdigest()
        context = result.context
        key = (context.tenant_id, context.space_id, context.document_id)
        db.execute("INSERT OR IGNORE INTO documents(tenant,space,document) VALUES (?,?,?)", key)
        row = db.execute("SELECT deleted FROM documents WHERE tenant=? AND space=? AND document=?",
                         key).fetchone()
        if row["deleted"]:
            raise StorageError("Документ удалён; обработка отменена")
        version_key = key + (context.document_version,)
        version = db.execute("SELECT checksum FROM versions WHERE tenant=? AND space=? AND document=? "
                             "AND version=?", version_key).fetchone()
        if version and version["checksum"] != context.checksum:
            raise StorageError("Содержимое существующей версии документа изменено")
        db.execute("INSERT OR IGNORE INTO versions VALUES (?,?,?,?,?)", version_key + (context.checksum,))
        existing = db.execute("SELECT digest FROM generations WHERE id=?", (result.generation_id,)).fetchone()
        if existing and existing["digest"] != digest:
            raise StorageError("Сохранённый checkpoint поколения неизменяем")
        db.execute("INSERT OR IGNORE INTO generations "
                   "(id,tenant,space,document,version,profile,result_json,digest,status) "
                   "VALUES (?,?,?,?,?,?,?,?, 'checkpoint')",
                   (result.generation_id,) + key + (context.document_version,
                                                   result.processing_profile_id, serialized, digest))
        # The beta exercises the atomic job/outbox contract without implementing Redis infrastructure.
        db.execute("INSERT OR IGNORE INTO outbox(job_id) VALUES (?)", (result.generation_id,))
        db.execute("UPDATE documents SET desired_version=MAX(desired_version,?) "
                   "WHERE tenant=? AND space=? AND document=?", (context.document_version,) + key)

    def load(self, generation_id):
        with self.connection() as db:
            row = db.execute("SELECT result_json FROM generations WHERE id=?", (generation_id,)).fetchone()
        if row is None:
            raise StorageError("Checkpoint не найден")
        return load_result(json.loads(row["result_json"]))

    def begin_index(self, generation_id):
        with self.connection() as db:
            row = db.execute("SELECT * FROM generations WHERE id=?", (generation_id,)).fetchone()
            if row is None:
                raise StorageError("Checkpoint не найден")
            key = (row["tenant"], row["space"], row["document"])
            document = db.execute("SELECT * FROM documents WHERE tenant=? AND space=? AND document=?",
                                  key).fetchone()
            if document["deleted"]:
                raise StorageError("Документ удалён")
            if row["status"] == "ready":
                return False
            if row["attempts"] >= 3:
                raise StorageError("Исчерпаны три попытки индексации")
            if row["version"] < document["desired_version"]:
                raise StorageError("Нельзя начать устаревшую версию вместо запрошенной новой")
            active = document["active_generation"]
            if active:
                active_version = db.execute("SELECT version FROM generations WHERE id=?", (active,)).fetchone()
                if active_version["version"] > row["version"]:
                    raise StorageError("Нельзя активировать устаревшую версию")
            db.execute("UPDATE documents SET desired_generation=?,desired_version=MAX(desired_version,?) "
                       "WHERE tenant=? AND space=? AND document=?", (generation_id, row["version"]) + key)
            db.execute("UPDATE generations SET status='indexing', attempts=attempts+1,error=NULL WHERE id=?",
                       (generation_id,))
            return True

    def activate(self, generation_id):
        with self.connection() as db:
            row = db.execute("SELECT * FROM generations WHERE id=?", (generation_id,)).fetchone()
            if row is None or row["status"] != "indexing":
                raise StorageError("Поколение не находится в состоянии indexing")
            key = (row["tenant"], row["space"], row["document"])
            document = db.execute("SELECT * FROM documents WHERE tenant=? AND space=? AND document=?",
                                  key).fetchone()
            if (document["deleted"] or document["desired_generation"] != generation_id
                    or row["version"] < document["desired_version"]):
                raise StorageError("Обработка отменена или заменена новым поколением")
            previous = document["active_generation"]
            db.execute("UPDATE documents SET active_generation=? WHERE tenant=? AND space=? AND document=?",
                       (generation_id,) + key)
            db.execute("UPDATE generations SET status='ready',error=NULL WHERE id=?", (generation_id,))
            return previous

    def fail(self, generation_id, error):
        with self.connection() as db:
            db.execute("UPDATE generations SET status='failed',error=? WHERE id=? AND status!='ready'",
                       (str(error), generation_id))

    def active_generations(self, tenant_id, space_id):
        with self.connection() as db:
            rows = db.execute("SELECT active_generation FROM documents WHERE tenant=? AND space=? "
                              "AND deleted=0 AND active_generation IS NOT NULL", (tenant_id, space_id)).fetchall()
        return [row[0] for row in rows]

    def delete_document(self, tenant_id, space_id, document_id):
        key = (tenant_id, space_id, document_id)
        with self.connection() as db:
            db.execute("INSERT OR IGNORE INTO documents(tenant,space,document) VALUES (?,?,?)", key)
            db.execute("UPDATE documents SET deleted=1,active_generation=NULL,desired_generation=NULL "
                       "WHERE tenant=? AND space=? AND document=?", key)
            rows = db.execute("SELECT id FROM generations WHERE tenant=? AND space=? AND document=?", key).fetchall()
        return [row[0] for row in rows]

    def pending_outbox(self):
        with self.connection() as db:
            return [row[0] for row in db.execute("SELECT job_id FROM outbox WHERE published=0")]

    def job_state(self, generation_id):
        with self.connection() as db:
            row = db.execute("SELECT g.status,g.attempts,d.deleted FROM generations g JOIN documents d "
                             "ON g.tenant=d.tenant AND g.space=d.space AND g.document=d.document "
                             "WHERE g.id=?", (generation_id,)).fetchone()
        return dict(row) if row else None

    def mark_published(self, job_id):
        with self.connection() as db:
            db.execute("UPDATE outbox SET published=1 WHERE job_id=?", (job_id,))

    def submit_processing(self, job_id, stored, context, profile):
        with self.connection() as db:
            key = (context.tenant_id, context.space_id, context.document_id)
            db.execute("INSERT OR IGNORE INTO documents(tenant,space,document) VALUES (?,?,?)", key)
            row = db.execute("SELECT deleted FROM documents WHERE tenant=? AND space=? AND document=?",
                             key).fetchone()
            if row["deleted"]:
                raise StorageError("Документ удалён; обработка отменена")
            db.execute("INSERT INTO processing_jobs(id,object_json,context_json,profile_json,status) "
                       "VALUES (?,?,?,?, 'pending')", (job_id, json.dumps(asdict(stored)),
                       json.dumps(asdict(context)), json.dumps(asdict(profile))))
            db.execute("UPDATE documents SET desired_version=MAX(desired_version,?) "
                       "WHERE tenant=? AND space=? AND document=?", (context.document_version,) + key)

    def claim_next_processing(self, lease_seconds=600, owner_id=None):
        if not isinstance(lease_seconds, (int, float)) or not 0 < lease_seconds < float("inf"):
            raise ValueError("lease_seconds must be positive and finite")
        now = time.time()
        if owner_id is None:
            owner_id = str(uuid4())
        elif not isinstance(owner_id, str) or not owner_id.strip():
            raise ValueError("owner_id must be a non-empty string")
        token = str(uuid4())
        with self.connection() as db:
            # Exhausted crashed jobs must leave running even without a worker
            # available to call fail_processing for the final attempt.
            db.execute("UPDATE processing_jobs SET status='failed',error='lease_exhausted',completed_at=?, "
                       "claim_token=NULL,owner_id=NULL,lease_until=NULL,claimed_at=NULL "
                       "WHERE attempts>=3 AND (status='pending' OR "
                       "(status='running' AND lease_until<=?))", (now, now))
            row = db.execute("SELECT * FROM processing_jobs WHERE attempts<3 AND "
                             "(status='pending' OR (status='running' AND lease_until<=?)) "
                             "ORDER BY rowid LIMIT 1", (now,)).fetchone()
            if row is None:
                return None
            db.execute("UPDATE processing_jobs SET status='running',attempts=attempts+1,claimed_at=?, "
                       "lease_until=?,claim_token=?,owner_id=?,completed_at=NULL,error=NULL WHERE id=?",
                       (now, now + lease_seconds, token, owner_id, row["id"]))
            return {"id": row["id"], "stored": json.loads(row["object_json"]),
                    "context": json.loads(row["context_json"]),
                    "profile": json.loads(row["profile_json"]), "claim_token": token,
                    "owner_id": owner_id, "lease_until": now + lease_seconds,
                    "attempt": row["attempts"] + 1}

    @staticmethod
    def _require_processing_claim(db, job_id, claim_token):
        row = db.execute("SELECT * FROM processing_jobs WHERE id=?", (job_id,)).fetchone()
        if (not isinstance(claim_token, str) or not claim_token or row is None or row["status"] != "running"
                or row["claim_token"] != claim_token or row["lease_until"] is None
                or row["lease_until"] <= time.time()):
            raise StorageError("Processing lease потеряна или истекла")
        return row

    def assert_processing_claim(self, job_id, claim_token):
        with self.connection() as db:
            self._require_processing_claim(db, job_id, claim_token)

    def heartbeat_processing(self, job_id, claim_token, lease_seconds=600):
        if not isinstance(lease_seconds, (int, float)) or not 0 < lease_seconds < float("inf"):
            raise ValueError("lease_seconds must be positive and finite")
        with self.connection() as db:
            self._require_processing_claim(db, job_id, claim_token)
            now = time.time()
            db.execute("UPDATE processing_jobs SET claimed_at=?,lease_until=? WHERE id=? AND claim_token=?",
                       (now, now + lease_seconds, job_id, claim_token))

    def save_processing_result(self, job_id, claim_token, result):
        with self.connection() as db:
            row = self._require_processing_claim(db, job_id, claim_token)
            expected = json.loads(row["context_json"])
            actual = asdict(result.context)
            fields = ("tenant_id", "space_id", "document_id", "document_version", "filename")
            stored = json.loads(row["object_json"])
            if (result.generation_id != job_id or any(actual[field] != expected[field] for field in fields)
                    or actual["checksum"] != stored["checksum"]):
                raise StorageError("Processing result не соответствует заявке")
            self._save_result(db, result)
            # JSON serialization/save may be expensive; no expired owner can
            # commit merely because it acquired the database lock in time.
            self._require_processing_claim(db, job_id, claim_token)

    def complete_processing(self, job_id, claim_token=None):
        with self.connection() as db:
            self._require_processing_claim(db, job_id, claim_token)
            db.execute("UPDATE processing_jobs SET status='complete',error=NULL,completed_at=?, "
                       "claim_token=NULL,owner_id=NULL,lease_until=NULL,claimed_at=NULL "
                       "WHERE id=? AND claim_token=?", (time.time(), job_id, claim_token))

    def fail_processing(self, job_id, error, retryable, claim_token=None):
        with self.connection() as db:
            row = self._require_processing_claim(db, job_id, claim_token)
            status = "pending" if retryable and row["attempts"] < 3 else "failed"
            db.execute("UPDATE processing_jobs SET status=?,error=?,claimed_at=NULL,claim_token=NULL, "
                       "owner_id=NULL,lease_until=NULL,completed_at=? WHERE id=? AND claim_token=?",
                       (status, str(error)[:1000], time.time() if status == "failed" else None,
                        job_id, claim_token))

    def processing_status(self, job_id):
        with self.connection() as db:
            row = db.execute("SELECT id,status,attempts,error,completed_at FROM processing_jobs WHERE id=?",
                             (job_id,)).fetchone()
        return dict(row) if row else None
