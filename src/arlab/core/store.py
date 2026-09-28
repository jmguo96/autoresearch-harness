"""Single-host transactional state, event log, budgets, and evidence-backed memory.

SQLite WAL is intentional for a textbook example. One OS lock owns a campaign;
workers write atomic artifacts, never the database. This is not a distributed queue.
"""
from __future__ import annotations

import contextlib
import math
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterator

from arlab.core.io import canonical, digest


class BudgetExceeded(RuntimeError):
    """The runtime, not the model, rejected a resource reservation."""


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, isolation_level=None, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events(
                seq INTEGER PRIMARY KEY AUTOINCREMENT, created REAL NOT NULL,
                kind TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS candidates(
                id TEXT PRIMARY KEY, parent TEXT, recipe TEXT NOT NULL,
                hypothesis TEXT NOT NULL, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS trials(
                id TEXT PRIMARY KEY, candidate TEXT NOT NULL REFERENCES candidates(id),
                status TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS actions(
                id INTEGER PRIMARY KEY AUTOINCREMENT, decision TEXT NOT NULL,
                status TEXT NOT NULL, feedback TEXT);
            CREATE TABLE IF NOT EXISTS reservations(
                id TEXT NOT NULL, resource TEXT NOT NULL, amount REAL NOT NULL,
                PRIMARY KEY(id, resource));
            CREATE TABLE IF NOT EXISTS usage(
                id TEXT NOT NULL, resource TEXT NOT NULL, amount REAL NOT NULL,
                PRIMARY KEY(id, resource));
            CREATE TABLE IF NOT EXISTS memories(
                id TEXT PRIMARY KEY, text TEXT NOT NULL, category TEXT NOT NULL,
                evidence TEXT NOT NULL, importance REAL NOT NULL, created REAL NOT NULL,
                active INTEGER NOT NULL DEFAULT 1, supersedes TEXT);
            CREATE TABLE IF NOT EXISTS artifacts(
                id TEXT PRIMARY KEY, kind TEXT NOT NULL, text TEXT NOT NULL,
                metadata TEXT NOT NULL, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS exposures(
                id INTEGER PRIMARY KEY AUTOINCREMENT, artifact TEXT NOT NULL,
                audience TEXT NOT NULL, split TEXT NOT NULL, created REAL NOT NULL);
        """)

    def close(self) -> None:
        self.db.close()

    @contextlib.contextmanager
    def transaction(self) -> Iterator[None]:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        else:
            self.db.execute("COMMIT")

    def get(self, key: str, default: Any = None) -> Any:
        import json
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def put(self, key: str, value: Any) -> None:
        self.db.execute("INSERT INTO meta VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        (key, canonical(value)))

    def bind(self, key: str, value: Any) -> None:
        previous = self.get(key)
        if previous is not None and canonical(previous) != canonical(value):
            raise ValueError(f"Immutable campaign binding changed: {key}; start a new campaign")
        self.put(key, value)

    def event(self, kind: str, payload: Any) -> str:
        cursor = self.db.execute("INSERT INTO events(created,kind,payload) VALUES (?,?,?)",
                                 (time.time(), kind, canonical(payload)))
        return f"event-{cursor.lastrowid}"

    def events(self, limit: int = 20) -> list[dict[str, Any]]:
        import json
        rows = self.db.execute("SELECT * FROM events ORDER BY seq DESC LIMIT ?", (limit,))
        return [{"id": f"event-{r['seq']}", "kind": r["kind"], "payload": json.loads(r["payload"])} for r in rows][::-1]

    def add_candidate(self, recipe: Any, hypothesis: Any, parent: str | None = None) -> str:
        cid = "c-" + digest(recipe)[:20]
        if parent and not self.candidate(parent):
            raise ValueError("Unknown parent candidate")
        self.db.execute("INSERT OR IGNORE INTO candidates VALUES (?,?,?,?,?)",
                        (cid, parent, canonical(recipe), canonical(hypothesis), time.time()))
        return cid

    def candidate(self, cid: str) -> dict[str, Any] | None:
        import json
        row = self.db.execute("SELECT * FROM candidates WHERE id=?", (cid,)).fetchone()
        if row is None:
            return None
        return {**dict(row), "recipe": json.loads(row["recipe"]), "hypothesis": json.loads(row["hypothesis"])}

    def candidates(self) -> list[dict[str, Any]]:
        return [self.candidate(r[0]) for r in self.db.execute("SELECT id FROM candidates ORDER BY created")]  # type: ignore[misc]

    def trial(self, tid: str) -> dict[str, Any] | None:
        import json
        row = self.db.execute("SELECT status,payload FROM trials WHERE id=?", (tid,)).fetchone()
        return {"status": row[0], **json.loads(row[1])} if row else None

    def save_trial(self, tid: str, candidate: str, status: str, payload: Any) -> None:
        self.db.execute("INSERT INTO trials VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE SET status=excluded.status,payload=excluded.payload",
                        (tid, candidate, status, canonical(payload)))

    def trials(self, status: str | None = None) -> list[dict[str, Any]]:
        import json
        rows = self.db.execute("SELECT status,payload FROM trials" + (" WHERE status=?" if status else ""),
                               (status,) if status else ())
        return [{"status": r[0], **json.loads(r[1])} for r in rows]

    def panel_trials(self) -> list[dict[str, Any]]:
        """Current-implementation evidence; historical trials remain queryable."""
        legacy = set((self.get("source_migration") or {}).get("legacy_trials", []))
        return [row for row in self.trials() if row.get("trial_id") not in legacy]

    def begin_action(self, decision: Any) -> int:
        cur = self.db.execute("INSERT INTO actions(decision,status) VALUES (?, 'pending')", (canonical(decision),))
        return int(cur.lastrowid)

    def pending_action(self) -> tuple[int, Any] | None:
        import json
        row = self.db.execute("SELECT id,decision FROM actions WHERE status='pending' ORDER BY id LIMIT 1").fetchone()
        return (row[0], json.loads(row[1])) if row else None

    def finish_action(self, aid: int, feedback: Any) -> None:
        self.db.execute("UPDATE actions SET status='done',feedback=? WHERE id=?", (canonical(feedback), aid))

    def count_actions(self) -> int:
        return int(self.db.execute("SELECT count(*) FROM actions").fetchone()[0])

    def totals(self, actual: bool = False) -> dict[str, float]:
        table = "usage" if actual else "reservations"
        return dict(self.db.execute(f"SELECT resource,sum(amount) FROM {table} GROUP BY resource"))

    def reserve(self, rid: str, requested: dict[str, float], limits: dict[str, float]) -> None:
        """Reserve worst-case cost before side effects; a failed attempt remains charged.

        Reusing an id is idempotent only when the amounts agree. Crash retries use a
        new attempt id, conservatively charging unknown work rather than hiding it.
        """
        with self.transaction():
            totals = self.totals()
            for resource, amount in requested.items():
                if not math.isfinite(amount) or amount < 0 or resource not in limits:
                    raise ValueError("Invalid or unbounded resource")
                old = self.db.execute("SELECT amount FROM reservations WHERE id=? AND resource=?", (rid, resource)).fetchone()
                if old:
                    if old[0] != amount:
                        raise ValueError("Reservation identity reused with different cost")
                    continue
                if totals.get(resource, 0) + amount > limits[resource]:
                    raise BudgetExceeded(f"{resource}: requested {amount}, reserved {totals.get(resource, 0)}, limit {limits[resource]}")
                self.db.execute("INSERT INTO reservations VALUES (?,?,?)", (rid, resource, amount))

    def record_usage(self, rid: str, amounts: dict[str, float]) -> None:
        for resource, amount in amounts.items():
            if not math.isfinite(amount) or amount < 0:
                raise ValueError("Invalid usage")
            self.db.execute("INSERT INTO usage VALUES (?,?,?) ON CONFLICT(id,resource) DO UPDATE SET amount=excluded.amount",
                            (rid, resource, amount))

    def evidence_exists(self, eid: str) -> bool:
        if eid.startswith("event-") and eid[6:].isdigit():
            return self.db.execute("SELECT 1 FROM events WHERE seq=?", (int(eid[6:]),)).fetchone() is not None
        return (self.trial(eid) is not None or
                self.db.execute("SELECT 1 FROM artifacts WHERE id=?", (eid,)).fetchone() is not None)

    def remember(self, text: str, category: str, evidence: list[str], *, importance: float = 0.6,
                 supersedes: str | None = None) -> str:
        if not evidence or not all(self.evidence_exists(e) for e in evidence):
            raise ValueError("Memory needs real, resolvable evidence references")
        mid = "m-" + digest({"text": text, "evidence": sorted(evidence), "category": category})[:20]
        with self.transaction():
            if supersedes:
                row = self.db.execute("SELECT 1 FROM memories WHERE id=?", (supersedes,)).fetchone()
                if not row:
                    raise ValueError("Unknown superseded memory")
                self.db.execute("UPDATE memories SET active=0 WHERE id=?", (supersedes,))
            self.db.execute("INSERT OR IGNORE INTO memories VALUES (?,?,?,?,?,?,1,?)",
                            (mid, text, category, canonical(evidence), importance, time.time(), supersedes))
        return mid

    def recall(self, query: str, limit: int = 5, *, categories: set[str] | None = None, require_overlap: bool = False) -> list[dict[str, Any]]:
        """Transparent lexical retrieval, recency, and importance; no vector service."""
        import json
        def tokens(text: str) -> set[str]:
            return set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text.lower()))
        requested = tokens(query)
        scored = []
        for row in self.db.execute("SELECT * FROM memories WHERE active=1"):
            if categories is not None and row["category"] not in categories:
                continue
            overlap = len(requested & tokens(row["text"])) / max(1, len(requested))
            if require_overlap and not overlap:
                continue
            recency = math.exp(-(time.time() - row["created"]) / (30 * 86400))
            score = 0.65 * overlap + 0.2 * row["importance"] + 0.15 * recency
            scored.append((score, {"id": row["id"], "text": row["text"], "category": row["category"],
                                   "evidence_ids": json.loads(row["evidence"]), "verified_claim": False}))
        return [r for _, r in sorted(scored, key=lambda x: x[0], reverse=True)[:limit]]

    def artifact(self, aid: str) -> dict[str, Any] | None:
        import json
        row = self.db.execute("SELECT * FROM artifacts WHERE id=?", (aid,)).fetchone()
        return {**dict(row), "metadata": json.loads(row["metadata"])} if row else None

    def save_artifact(self, kind: str, text: str, metadata: dict[str, Any]) -> str:
        """Immutable snapshot. URI/title alone never identifies the bytes we read."""
        aid = kind + "-" + digest({"text": text, "metadata": metadata})[:24]
        self.db.execute("INSERT OR IGNORE INTO artifacts VALUES (?,?,?,?,?)",
                        (aid, kind, text, canonical(metadata), time.time()))
        return aid

    def artifact_index(self, kind: str, limit: int = 10, offset: int = 0) -> list[dict[str, Any]]:
        import json
        rows = self.db.execute("SELECT id,metadata,created,length(text) AS characters FROM artifacts "
                               "WHERE kind=? ORDER BY created DESC,id LIMIT ? OFFSET ?", (kind, limit, offset))
        return [{**dict(r), "metadata": json.loads(r["metadata"])} for r in rows]

    def expose(self, artifact: str, audience: str, split: str) -> None:
        self.db.execute("INSERT INTO exposures(artifact,audience,split,created) VALUES (?,?,?,?)",
                        (artifact, audience, split, time.time()))


@contextlib.contextmanager
def campaign_lock(root: Path) -> Iterator[None]:
    """Fail fast rather than letting two agents mutate one campaign concurrently."""
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".campaign.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process owns this campaign") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
