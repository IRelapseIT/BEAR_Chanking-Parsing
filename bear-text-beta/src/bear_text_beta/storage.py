import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from .errors import StorageError
from .serialization import load_result


class SQLiteManifest:
    """Local job/checkpoint/activation state, replaceable with the team's PostgreSQL adapter."""
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript("""
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
            """)

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
        serialized = json.dumps(asdict(result), ensure_ascii=False, sort_keys=True, allow_nan=False)
        digest = hashlib.sha256(serialized.encode()).hexdigest()
        context = result.context
        key = (context.tenant_id, context.space_id, context.document_id)
        with self.connection() as db:
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
            active = document["active_generation"]
            if active:
                active_version = db.execute("SELECT version FROM generations WHERE id=?", (active,)).fetchone()
                if active_version["version"] > row["version"]:
                    raise StorageError("Нельзя активировать устаревшую версию")
            db.execute("UPDATE documents SET desired_generation=? WHERE tenant=? AND space=? AND document=?",
                       (generation_id,) + key)
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
            if document["deleted"] or document["desired_generation"] != generation_id:
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
            db.execute("INSERT INTO processing_jobs(id,object_json,context_json,profile_json,status) "
                       "VALUES (?,?,?,?, 'pending')", (job_id, json.dumps(asdict(stored)),
                       json.dumps(asdict(context)), json.dumps(asdict(profile))))

    def claim_next_processing(self, lease_seconds=600):
        import time
        now = time.time()
        with self.connection() as db:
            row = db.execute("SELECT * FROM processing_jobs WHERE status='pending' OR "
                             "(status='running' AND claimed_at<? AND attempts<3) "
                             "ORDER BY rowid LIMIT 1", (now - lease_seconds,)).fetchone()
            if row is None:
                return None
            db.execute("UPDATE processing_jobs SET status='running',attempts=attempts+1,claimed_at=? "
                       "WHERE id=?", (now, row["id"]))
            return {"id": row["id"], "stored": json.loads(row["object_json"]),
                    "context": json.loads(row["context_json"]),
                    "profile": json.loads(row["profile_json"])}

    def complete_processing(self, job_id):
        with self.connection() as db:
            db.execute("UPDATE processing_jobs SET status='complete',error=NULL WHERE id=?", (job_id,))

    def fail_processing(self, job_id, error, retryable):
        with self.connection() as db:
            row = db.execute("SELECT attempts FROM processing_jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                return
            status = "pending" if retryable and row["attempts"] < 3 else "failed"
            db.execute("UPDATE processing_jobs SET status=?,error=?,claimed_at=NULL WHERE id=?",
                       (status, str(error)[:1000], job_id))

    def processing_status(self, job_id):
        with self.connection() as db:
            row = db.execute("SELECT id,status,attempts,error FROM processing_jobs WHERE id=?",
                             (job_id,)).fetchone()
        return dict(row) if row else None
