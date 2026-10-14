#!/usr/bin/env python3
"""Owner maintenance: rebind one paused local study after control-only source edits.

Verified historical scores stay in the panel; their checkpoints retain the old
source identity and are not continuable. Run dry-run first. Applying creates
backups in the existing deployment directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
import uuid
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
APPROVED_SOURCE_SHA256 = "a3083ebd49430516d92d31d7aabebb12fde35d1a143b6229832977ada9c46012"
APPROVED_TEMPLATE_FILES = {
    "src/arlab/algorithms/templates.py": "ec37a7117d94db8bf5540f2df739bd3311779a5334a22435d6565e8ab0576564",
    "src/arlab/templates/iql_candidate.py": "1736a4cf84a0565ad32f520ce0c767e4c53857af16a634f397d1588ef1a3f211",
}
sys.path.insert(0, str(SOURCE / "src"))

from arlab.core.io import canonical, read_json, source_digest, write_json  # noqa: E402
from arlab.core.checkpoints import training_binding  # noqa: E402
from arlab.core.store import campaign_lock  # noqa: E402
from arlab.core.config import AgentConfig  # noqa: E402
from arlab.deployment.manager import _source_identity  # noqa: E402


def critical_source_unchanged() -> int:
    """Check training, evaluation, policy and audit files against the release manifest."""
    manifest = read_json(SOURCE / "FILE_MANIFEST.json")
    expected = {item["path"]: item["sha256"] for item in manifest["files"]}
    expected.update(APPROVED_TEMPLATE_FILES)
    prefixes = ("src/arlab/tasks/", "src/arlab/algorithms/")
    exact = {
        "src/arlab/core/checkpoints.py", "src/arlab/core/panels.py",
        "src/arlab/core/stats.py", "src/arlab/core/jobs.py",
        "src/arlab/worker.py", "src/arlab/audit.py",
    }
    paths = [path for path in expected if path.startswith(prefixes) or path in exact or path in APPROVED_TEMPLATE_FILES]
    for relative in paths:
        current = hashlib.sha256((SOURCE / relative).read_bytes()).hexdigest()
        if current != expected[relative]:
            raise ValueError(f"Training/evaluation source changed: {relative}; in-place rebind refused")
    return len(paths)


def migrate(expected_old: str, apply: bool) -> dict:
    root = SOURCE / ".arlab" / "d4rl9"
    study = root / "runs" / "study" / "development"
    deployment_path, checks_path = root / "deployment.json", root / "checks.json"
    database_path = study / "state.sqlite3"
    current_source = source_digest()
    if current_source != APPROVED_SOURCE_SHA256:
        raise ValueError("This one-time control update is not approved for the current source")
    current_identity = _source_identity(SOURCE)
    checked_files = critical_source_unchanged()
    with campaign_lock(study):
        deployment = read_json(deployment_path)
        checks = read_json(checks_path)
        if deployment.get("status") != "ready" or checks.get("status") != "passed":
            raise ValueError("Deployment and smoke checks must be ready")
        db = sqlite3.connect(database_path, isolation_level=None)
        try:
            get = lambda key: db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
            old_row = get("source_sha256")
            status_row = get("status")
            agent_row = get("agent")
            prior_row = get("source_migration")
            prior = json.loads(prior_row[0]) if prior_row else {}
            if not old_row or json.loads(old_row[0]) != expected_old:
                raise ValueError("Study source does not match expected old source")
            if checks.get("source_sha256") != expected_old:
                raise ValueError("Deployment checks do not match the study source")
            if not status_row or json.loads(status_row[0]) not in ("paused", "running", "model_failed", "blocked") or get("frozen"):
                raise ValueError("Only an unfrozen stopped study can be rebound")
            if not agent_row:
                raise ValueError("Study agent configuration is missing")
            old_agent = json.loads(agent_row[0])
            new_agent = AgentConfig.model_validate({**old_agent, "instruction": AgentConfig().instruction})
            pending = db.execute("SELECT id,decision FROM actions WHERE status='pending'").fetchall()
            if len(pending) > 1:
                raise ValueError("Multiple pending actions; manual recovery required")
            interrupted = None
            if pending:
                action_id, decision_json = pending[0]
                decision = json.loads(decision_json)
                action = decision.get("action", {})
                if action.get("kind") != "run":
                    raise ValueError("Pending non-training action must settle before source rebind")
                matches = []
                for trial_id, payload in db.execute("SELECT id,payload FROM trials WHERE status='running'"):
                    trial = json.loads(payload)
                    if all(trial.get(key) == action.get(key) for key in (
                            "candidate", "case", "train_seed", "steps")):
                        matches.append((trial_id, trial))
                if len(matches) != 1:
                    raise ValueError("Pending training action has no unique running trial")
                trial_id, trial = matches[0]
                if (study / "jobs" / trial_id / "train" / "training.json").exists():
                    raise ValueError("Pending training has a completed receipt; recover it before migration")
                interrupted = {"action_id": action_id, "trial_id": trial_id, "trial": trial}
            rows = db.execute("SELECT id,payload FROM trials").fetchall()
            incompatible_checkpoints = [row[0] for row in rows]
            legacy_trials = prior.get("legacy_trials", [])
            trial_sources = {}
            for trial_id, payload in rows:
                trial = json.loads(payload)
                request = study / "jobs" / trial_id / "train-request.json"
                if not request.is_file():
                    raise ValueError(f"Historical trial source is not verified: {trial_id}")
                train_request = read_json(request)
                recorded_source = train_request.get("source_sha256")
                request_binding = training_binding(train_request)
                if (not isinstance(recorded_source, str) or len(recorded_source) != 64
                        or (trial.get("request_sha256") != request_binding
                            and not (interrupted and trial_id == interrupted["trial_id"]
                                     and train_request.get("recipe") is not None))):
                    raise ValueError(f"Historical trial request binding is not verified: {trial_id}")
                trial_sources[trial_id] = recorded_source
            summary = {
                "root": str(root), "study": str(study),
                "old_source_sha256": expected_old,
                "new_source_sha256": current_source,
                "legacy_trials": legacy_trials,
                "incompatible_checkpoint_trials": incompatible_checkpoints,
                "legacy_trial_sources": trial_sources,
                "critical_release_files_verified": checked_files,
                "effect": "Keep verified scores in the panel; old checkpoints are not continuable; new seeds start fresh.",
                "agent_policy_updated": old_agent.get("instruction") != new_agent.instruction,
                "interrupted_action": {key: interrupted[key] for key in ("action_id", "trial_id")} if interrupted else None,
                "writes": apply,
            }
            if not apply:
                return summary
            marker = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
            backup_db = study / f"state.before-control-update-{marker}.sqlite3"
            backup_deployment = root / f"deployment.before-control-update-{marker}.json"
            backup_checks = root / f"checks.before-control-update-{marker}.json"
            agent_path = study / "agent.json"
            backup_agent = study / f"agent.before-control-update-{marker}.json"
            for target in (backup_db, backup_deployment, backup_checks, backup_agent):
                if target.exists() or target.is_symlink():
                    raise ValueError(f"Backup target already exists: {target}")
            with sqlite3.connect(backup_db) as backup:
                db.backup(backup)
            backup_db.chmod(0o600)
            write_json(backup_deployment, deployment)
            write_json(backup_checks, checks)
            write_json(backup_agent, read_json(agent_path))
            migration = {
                "old_source_sha256": expected_old,
                "new_source_sha256": current_source,
                "lineage": prior.get("lineage", [{"old": prior["old_source_sha256"], "new": prior["new_source_sha256"]}]
                                     if prior else []) + [{"old": expected_old, "new": current_source}],
                "legacy_trials": legacy_trials,
                "incompatible_checkpoint_trials": incompatible_checkpoints,
                "legacy_trial_sources": trial_sources,
                "checkpoint_policy": "Verified historical scores remain in the panel; old checkpoints cannot continue. Run a new trajectory.",
                "critical_release_files_verified": checked_files,
                "backup": backup_db.name,
                "control_policy": "Proposal approval uses review_id, parent and verdict. IQL expression parents now expose candidate.py/networks.py and edit_algorithm creates an iql-source copy with the original config; prior missing-source errors are resolved. Inspect tool arguments by target='tool'; schema IDs are recipe/decision. Model-response failures and tool failures have separate counters.",
            }
            deployment["source_identity"] = current_identity
            deployment["control_source_migration"] = migration
            checks["source_sha256"] = current_source
            checks["control_source_migration"] = migration
            write_json(deployment_path, deployment)
            write_json(checks_path, checks)
            write_json(agent_path, new_agent)
            db.execute("BEGIN IMMEDIATE")
            try:
                if interrupted:
                    trial_id = interrupted["trial_id"]
                    action_id = interrupted["action_id"]
                    reason = "Training was interrupted before control-policy migration; partial artifacts and reservations are retained."
                    failed = {**interrupted["trial"], "error": reason, "error_type": "InterruptedForMigration"}
                    feedback = {"status": "infrastructure_failure", "summary": reason,
                                "evidence_ids": [], "details": {"last_failed_trial": trial_id}}
                    db.execute("UPDATE trials SET status='failed',payload=? WHERE id=?", (canonical(failed), trial_id))
                    db.execute("UPDATE actions SET status='done',feedback=? WHERE id=?", (canonical(feedback), action_id))
                    db.execute("INSERT INTO meta(key,value) VALUES ('last_feedback',?) "
                               "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (canonical(feedback),))
                    db.execute("INSERT INTO meta(key,value) VALUES ('last_failed_trial',?) "
                               "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (canonical(trial_id),))
                    db.execute("INSERT INTO events(created,kind,payload) VALUES (?,?,?)",
                               (time.time(), "trial_interrupted_for_migration", canonical({
                                   "action_id": action_id, "trial_id": trial_id,
                               })))
                db.execute("UPDATE meta SET value=? WHERE key='source_sha256'", (canonical(current_source),))
                db.execute("UPDATE meta SET value=? WHERE key='agent'", (canonical(new_agent),))
                review_row = get("proposal_review")
                if review_row:
                    review = json.loads(review_row[0])
                    if review and "hypothesis_sha256" in review:
                        event_id = review.get("id", "")
                        if not event_id.startswith("event-") or not event_id[6:].isdigit():
                            raise ValueError("Invalid persisted review reference")
                        event = db.execute("SELECT payload FROM events WHERE seq=? AND kind='proposal_reviewed'",
                                           (int(event_id[6:]),)).fetchone()
                        if not event:
                            raise ValueError("Persisted proposal review event is missing")
                        review["hypothesis"] = json.loads(event[0])["hypothesis"]
                        review.pop("hypothesis_sha256")
                        db.execute("UPDATE meta SET value=? WHERE key='proposal_review'", (canonical(review),))
                db.execute("INSERT INTO meta(key,value) VALUES ('status',?) "
                           "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (canonical("paused"),))
                db.execute("INSERT INTO meta(key,value) VALUES ('source_migration',?) "
                           "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (canonical(migration),))
                db.execute("INSERT INTO events(created,kind,payload) VALUES (?,?,?)",
                           (time.time(), "control_source_migrated", canonical(migration)))
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
            summary["backups"] = [str(backup_db), str(backup_deployment), str(backup_checks), str(backup_agent)]
            return summary
        finally:
            db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-old-source", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(migrate(args.expected_old_source, args.apply), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
