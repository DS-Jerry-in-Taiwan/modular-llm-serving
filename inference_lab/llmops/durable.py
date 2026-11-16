"""Small, durable job ledger for the local V2 service.

The ledger is deliberately boring: SQLite is used for atomic transactions, and
the ``events`` table is append-only.  The mutable row in ``jobs`` is only an
index of the last event.  Evidence is never copied or rewritten by this module;
terminal records contain a path and a digest and are rejected when that raw
file is absent or has changed.

This is a local durability seam, not a distributed queue.  On restart,
``recover()`` applies the explicit ``fail_incomplete_v1`` policy to every
queued/running job.  Recovery is itself a terminal event and is idempotent.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterator


SCHEMA_VERSION = 1
RECOVERY_POLICY = "fail_incomplete_v1"
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
ACTIVE_STATES = frozenset({"queued", "running"})
TERMINAL_STATES = frozenset({
    "completed", "succeeded", "failed", "cancelled", "deadline_exceeded",
})
ALL_STATES = ACTIVE_STATES | TERMINAL_STATES


class DurableJobError(ValueError):
    """Base class for fail-closed ledger errors."""


class JobConflictError(DurableJobError):
    """An ID was reused for a different job or terminal record."""


class InvalidTransitionError(DurableJobError):
    """A state transition is not legal."""


class ProvenanceError(DurableJobError):
    """Evidence is missing, malformed, or no longer matches its digest."""


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise DurableJobError(f"{name} must be a bounded identifier")
    return value


def _json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise DurableJobError("record metadata must be JSON-safe") from exc


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _sha256(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except (OSError, ValueError) as exc:
        raise ProvenanceError(f"evidence is unavailable: {path}") from exc
    return digest.hexdigest()


@dataclass(frozen=True)
class EvidenceRef:
    """A reference to immutable raw evidence owned by the caller."""

    path: str
    sha256: str
    source_commit: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise ProvenanceError("evidence path is required")
        if not _SHA256.fullmatch(self.sha256):
            raise ProvenanceError("evidence sha256 is invalid")
        if not _COMMIT.fullmatch(self.source_commit):
            raise ProvenanceError("evidence source_commit is invalid")

    def verify(self) -> None:
        path = Path(self.path)
        if not path.is_file():
            raise ProvenanceError(f"evidence is missing: {self.path}")
        actual = _sha256(path)
        if actual != self.sha256:
            raise ProvenanceError(f"evidence digest mismatch: {self.path}")

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256, "source_commit": self.source_commit}

    @classmethod
    def from_value(cls, value: Any) -> "EvidenceRef":
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise ProvenanceError("evidence must be an object")
        try:
            ref = cls(path=value["path"], sha256=value["sha256"], source_commit=value["source_commit"])
        except (KeyError, TypeError) as exc:
            raise ProvenanceError("evidence requires path, sha256, and source_commit") from exc
        ref.verify()
        return ref


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    request_id: str
    run_id: str
    state: str
    created_at: str
    updated_at: str
    metadata: dict[str, Any]
    terminal_record: dict[str, Any] | None = None

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES


@dataclass(frozen=True)
class JobEvent:
    event_id: str
    job_id: str
    request_id: str
    run_id: str
    from_state: str | None
    to_state: str
    reason: str
    created_at: str
    payload: dict[str, Any]


class DurableJobStore:
    """SQLite-backed append-only state transitions.

    ``create_job`` and ``transition`` are safe to call repeatedly after a
    process restart.  A repeated terminal call with the same state and exact
    record returns the existing record; any competing terminal record raises
    :class:`JobConflictError`.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA busy_timeout = 30000")
        if str(self.path) != ":memory:":
            self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.commit()
        self._init_schema()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "DurableJobStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _init_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                request_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                state TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                terminal_event_id TEXT
            );
            CREATE UNIQUE INDEX IF NOT EXISTS jobs_request_id ON jobs(request_id);
            CREATE UNIQUE INDEX IF NOT EXISTS jobs_run_id ON jobs(run_id);
            CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                job_id TEXT NOT NULL REFERENCES jobs(job_id),
                request_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                from_state TEXT,
                to_state TEXT NOT NULL,
                reason TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS events_job ON events(job_id, sequence);
            """
        )
        self._connection.commit()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            yield self._connection
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

    @staticmethod
    def _event_payload(reason: str, terminal: dict[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "reason": reason}
        if terminal is not None:
            payload["terminal_record"] = terminal
        return payload

    def _insert_event(self, db: sqlite3.Connection, *, job: sqlite3.Row,
                      from_state: str | None, to_state: str, reason: str,
                      payload: dict[str, Any], event_id: str) -> str:
        encoded = _json(payload)
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        now = _now()
        db.execute(
            "INSERT INTO events(event_id,job_id,request_id,run_id,from_state,to_state,reason,payload_json,payload_sha256,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (event_id, job["job_id"], job["request_id"], job["run_id"], from_state, to_state, reason, encoded, digest, now),
        )
        db.execute("UPDATE jobs SET state=?,updated_at=?,terminal_event_id=? WHERE job_id=?",
                   (to_state, now, event_id if to_state in TERMINAL_STATES else None, job["job_id"]))
        return now

    def create_job(self, request_id: str, job_id: str, run_id: str,
                   *, metadata: dict[str, Any] | None = None) -> JobRecord:
        request_id = _identifier(request_id, "request_id")
        job_id = _identifier(job_id, "job_id")
        run_id = _identifier(run_id, "run_id")
        metadata = {} if metadata is None else dict(metadata)
        encoded_metadata = _json(metadata)
        with self._write() as db:
            existing = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if existing:
                if (existing["request_id"], existing["run_id"], existing["metadata_json"]) != (request_id, run_id, encoded_metadata):
                    raise JobConflictError(f"job_id already exists: {job_id}")
                return self._record(existing)
            for column, value in (("request_id", request_id), ("run_id", run_id)):
                conflict = db.execute(f"SELECT job_id FROM jobs WHERE {column}=?", (value,)).fetchone()
                if conflict:
                    raise JobConflictError(f"{column} already belongs to {conflict['job_id']}")
            now = _now()
            db.execute("INSERT INTO jobs(job_id,request_id,run_id,state,metadata_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                       (job_id, request_id, run_id, "queued", encoded_metadata, now, now))
            job = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            self._insert_event(db, job=job, from_state=None, to_state="queued", reason="created",
                               payload=self._event_payload("created"), event_id=f"create:{job_id}")
            return self._record(db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone())

    # A concise alias is useful to adapters that call this a ledger.
    create = create_job

    def _record(self, row: sqlite3.Row) -> JobRecord:
        if row["state"] not in ALL_STATES:
            raise DurableJobError("job contains an unknown state")
        if row["state"] in TERMINAL_STATES and not row["terminal_event_id"]:
            raise DurableJobError("terminal job is missing its terminal event")
        if row["state"] not in TERMINAL_STATES and row["terminal_event_id"]:
            raise DurableJobError("active job unexpectedly has a terminal event")
        terminal: dict[str, Any] | None = None
        if row["terminal_event_id"]:
            event = self._connection.execute("SELECT * FROM events WHERE event_id=?", (row["terminal_event_id"],)).fetchone()
            if event is None:
                raise DurableJobError("terminal event is missing")
            if event["job_id"] != row["job_id"] or event["request_id"] != row["request_id"] or event["run_id"] != row["run_id"] or event["to_state"] != row["state"]:
                raise DurableJobError("terminal event identity does not match job")
            payload = json.loads(event["payload_json"])
            if hashlib.sha256(event["payload_json"].encode()).hexdigest() != event["payload_sha256"]:
                raise DurableJobError("event payload digest mismatch")
            terminal = payload.get("terminal_record")
            if terminal and terminal.get("evidence"):
                EvidenceRef.from_value(terminal["evidence"]).verify()
        return JobRecord(row["job_id"], row["request_id"], row["run_id"], row["state"], row["created_at"], row["updated_at"], json.loads(row["metadata_json"]), terminal)

    def get(self, job_id: str) -> JobRecord | None:
        job_id = _identifier(job_id, "job_id")
        row = self._connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return self._record(row) if row else None

    def history(self, job_id: str) -> list[JobEvent]:
        job_id = _identifier(job_id, "job_id")
        rows = self._connection.execute("SELECT * FROM events WHERE job_id=? ORDER BY sequence", (job_id,)).fetchall()
        result = []
        for row in rows:
            if hashlib.sha256(row["payload_json"].encode()).hexdigest() != row["payload_sha256"]:
                raise DurableJobError("event payload digest mismatch")
            result.append(JobEvent(row["event_id"], row["job_id"], row["request_id"], row["run_id"], row["from_state"], row["to_state"], row["reason"], row["created_at"], json.loads(row["payload_json"])))
        return result

    def transition(self, job_id: str, to_state: str, *, reason: str = "",
                   evidence: EvidenceRef | dict[str, Any] | None = None,
                   provenance: dict[str, Any] | None = None,
                   event_id: str | None = None) -> JobRecord:
        job_id = _identifier(job_id, "job_id")
        if to_state not in ALL_STATES or to_state == "queued":
            raise InvalidTransitionError(f"invalid target state: {to_state}")
        if not isinstance(reason, str) or len(reason) > 500:
            raise DurableJobError("reason must be at most 500 characters")
        ref = EvidenceRef.from_value(evidence) if evidence is not None else None
        if to_state in TERMINAL_STATES and to_state in {"completed", "succeeded"} and ref is None:
            raise ProvenanceError("successful terminal state requires evidence")
        terminal: dict[str, Any] | None = None
        if to_state in TERMINAL_STATES:
            if ref is not None:
                ref.verify()
            terminal = {"schema_version": SCHEMA_VERSION, "state": to_state, "reason": reason}
            if ref is not None:
                terminal["evidence"] = ref.to_dict()
            terminal["provenance"] = dict(provenance or ({"source_commit": ref.source_commit} if ref else {}))
            if ref is not None and terminal["provenance"].get("source_commit") != ref.source_commit:
                raise ProvenanceError("terminal provenance source_commit does not match evidence")
            _json(terminal)
        with self._write() as db:
            row = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise DurableJobError(f"unknown job: {job_id}")
            current = row["state"]
            if current in TERMINAL_STATES:
                if current == to_state and terminal is not None:
                    existing = self._record(row)
                    if existing.terminal_record == terminal:
                        return existing
                raise JobConflictError(f"job already terminal: {job_id}")
            allowed = {"queued": {"running", *TERMINAL_STATES}, "running": set(TERMINAL_STATES)}[current]
            if to_state not in allowed:
                raise InvalidTransitionError(f"cannot transition {current} -> {to_state}")
            event_id = event_id or (f"terminal:{job_id}:{to_state}" if to_state in TERMINAL_STATES else f"state:{job_id}:{to_state}")
            if not isinstance(event_id, str) or not event_id:
                raise DurableJobError("event_id must be non-empty")
            payload = self._event_payload(reason, terminal)
            self._insert_event(db, job=row, from_state=current, to_state=to_state, reason=reason, payload=payload, event_id=event_id)
            return self._record(db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone())

    record_terminal = transition

    def recover(self) -> list[JobRecord]:
        """Apply the deterministic ``fail_incomplete_v1`` restart policy."""
        rows = self._connection.execute("SELECT job_id FROM jobs WHERE state IN ('queued','running') ORDER BY created_at,job_id").fetchall()
        recovered: list[JobRecord] = []
        for row in rows:
            recovered.append(self.transition(row["job_id"], "failed", reason=f"restart recovery ({RECOVERY_POLICY})"))
        return recovered

    recover_incomplete = recover
