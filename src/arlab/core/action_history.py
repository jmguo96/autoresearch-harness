"""Deterministic action-history views and loop guards for model-driven research."""
from __future__ import annotations

import json
from typing import Any

from arlab.core.io import canonical, digest


PROGRESS_KINDS = {
    "run", "continue_trial", "check_candidate", "review_proposal", "propose", "edit_algorithm",
    "remember", "set_branch_status", "finish",
}
JOURNAL_VERSION = 2
JOURNAL_MILESTONE_LIMIT = 200
OBSERVATION_KINDS = {
    "plan", "inspect", "read_trial_log", "query_experiments", "compare",
    "search_literature", "search_web", "fetch_source", "read_source",
    "find_source", "search_repositories", "read_repository", "recall",
}


def observation_key(action: dict[str, Any]) -> str | None:
    """Identify immutable/read-through observations while ignoring display limits."""
    kind = action.get("kind")
    identity: dict[str, Any] | None = None
    if kind == "inspect":
        target = action.get("target")
        # An empty feedback ID and the plan are mutable aliases; do not cache them.
        if target == "plan" or target == "feedback" and not action.get("id"):
            return None
        identity = {k: action.get(k) for k in (
            "kind", "target", "id", "path", "start_line", "offset",
        )}
    elif kind == "read_trial_log":
        # metric_summary now always covers every numeric field, so changing the
        # display field list does not create a new observation.
        identity = {k: action.get(k) for k in (
            "kind", "trial", "stream", "start_step", "end_step", "tail",
            "start_line", "offset",
        )}
    elif kind == "read_source":
        identity = {k: action.get(k) for k in ("kind", "source", "start_line", "offset")}
    elif kind == "find_source":
        identity = {k: action.get(k) for k in ("kind", "source", "query", "offset")}
    elif kind == "read_repository":
        identity = {k: action.get(k) for k in ("kind", "repository", "ref", "path", "offset")}
    elif kind in ("search_literature", "search_repositories") and not action.get("refresh"):
        identity = {k: v for k, v in action.items() if k not in ("limit", "max_characters")}
    elif kind == "fetch_source" and not action.get("refresh"):
        identity = {"kind": kind, "url": action.get("url")}
    if identity is None:
        return None
    return kind + "-" + digest(identity)[:20]


def successful_observation(store, key: str) -> tuple[int, int] | None:
    """Return the first matching action ID and total successful read count."""
    journal = store.get("research_journal")
    if isinstance(journal, dict) and journal.get("version") == JOURNAL_VERSION:
        receipt = journal.get("observations", {}).get(key)
        if receipt:
            return int(receipt["first_action_id"]), int(receipt["successful_reads"])
    first = None
    count = 0
    rows = store.db.execute(
        "SELECT id,decision,feedback FROM actions WHERE status='done' ORDER BY id"
    )
    for row in rows:
        decision = json.loads(row["decision"])
        feedback = json.loads(row["feedback"]) if row["feedback"] else {}
        if feedback.get("status") == "ok" and observation_key(decision.get("action", {})) == key:
            first = first or int(row["id"])
            count += 1
    return (first, count) if first is not None else None


def observation_streak(store) -> int:
    """Count decisions since the last action that could advance research state."""
    kinds = sorted(PROGRESS_KINDS)
    placeholders = ",".join("?" for _ in kinds)
    row = store.db.execute(
        "SELECT COUNT(*) FROM actions WHERE status='done' AND id > COALESCE(("
        "SELECT MAX(id) FROM actions WHERE status='done' "
        f"AND json_extract(decision,'$.action.kind') IN ({placeholders})"
        "),0)",
        kinds,
    ).fetchone()
    return int(row[0])


def _result_slot(action: dict[str, Any], key: str) -> str:
    """Group alternate views of the same resource into one context slot."""
    if action.get("kind") == "read_trial_log":
        return canonical({k: action.get(k) for k in ("kind", "trial", "stream")})
    if action.get("kind") == "inspect":
        return canonical({k: action.get(k) for k in ("kind", "target", "id", "path")})
    return key


def _details_view(details: dict[str, Any], limit: int) -> dict[str, Any]:
    handles = {}
    for key in (
        "target", "id", "path", "trial", "stream", "source", "span", "provider",
        "start_line", "end_line", "total_lines", "file_lines", "filtered_lines",
        "next", "next_offset", "truncated", "metric_summary", "total_results",
    ):
        if key in details:
            handles[key] = details[key]
    if isinstance(details.get("items"), list):
        handles["items_preview"] = details["items"][:3]
    if "text" in details:
        text = str(details["text"])
        handles["text_sha256"] = digest(text)
        try:
            structured = json.loads(text)
        except json.JSONDecodeError:
            structured = None
        if structured is not None and len(canonical(structured)) <= max(300, limit - 500):
            handles["content"] = structured
        else:
            handles["text_preview"] = text[:max(300, limit // 2)]
    if not handles:
        handles["preview"] = canonical(details)[:limit]
    while len(canonical(handles)) > limit and handles.get("items_preview"):
        handles["items_preview"].pop()
    if len(canonical(handles)) > limit and "text_preview" in handles:
        handles["text_preview"] = handles["text_preview"][:max(200, limit // 4)] + "…"
    if len(canonical(handles)) > limit:
        return {"preview": canonical(handles)[:limit] + "…"}
    return handles


def recent_tool_results(store, *, limit: int, count: int = 4) -> list[dict[str, Any]]:
    """Keep recent distinct observations so alternating tools cannot erase each other."""
    rows = store.db.execute(
        "SELECT id,decision,feedback FROM actions "
        "WHERE status='done' AND feedback IS NOT NULL ORDER BY id DESC LIMIT 100"
    )
    selected = []
    seen: set[str] = set()
    for row in rows:
        decision = json.loads(row["decision"])
        action = decision.get("action", {})
        key = observation_key(action)
        feedback = json.loads(row["feedback"])
        slot = _result_slot(action, key) if key else None
        if key is None or slot in seen or feedback.get("status") != "ok":
            continue
        seen.add(slot)
        selected.append((row, action, key, feedback))
        if len(selected) >= count:
            break
    if not selected:
        return []
    each = max(700, limit // len(selected))
    result = []
    for row, action, key, feedback in selected:
        result.append({
            "action_id": int(row["id"]),
            "observation_key": key,
            "action": action,
            "summary": feedback.get("summary", ""),
            "evidence_ids": feedback.get("evidence_ids", [])[:5],
            "result": _details_view(feedback.get("details", {}), each),
            "instruction": "Already read successfully; do not request the same observation again.",
        })
    return result


def _action_view(action: dict[str, Any]) -> dict[str, Any]:
    kind = action.get("kind")
    keys = (
        "kind", "candidate", "parent", "trial", "case", "train_seed", "steps",
        "eval_episodes", "target", "id", "path", "source", "query", "provider",
        "active", "category", "new_branch", "template",
    )
    view = {key: action[key] for key in keys if key in action and action[key] not in (None, "")}
    if kind == "plan":
        view["reason"] = str(action.get("reason", ""))[:500]
        view["items"] = [
            {
                "id": item.get("id"),
                "objective": str(item.get("objective", ""))[:300],
                "state": item.get("state", "pending"),
            }
            for item in action.get("items", [])[:12]
        ]
    if kind == "remember":
        view["text"] = str(action.get("text", ""))[:800]
        view["evidence_ids"] = action.get("evidence_ids", [])[:10]
    return view


def _outcome_view(feedback: dict[str, Any]) -> dict[str, Any]:
    details = feedback.get("details", {})
    keys = (
        "trial_id", "candidate", "case", "train_seed", "steps", "score", "metric",
        "continued_from", "training_reused_from", "branch", "active", "comparison",
        "source", "span", "search_ref", "next_offset", "reason",
    )
    outcome = {key: details[key] for key in keys if key in details}
    if isinstance(details.get("items"), list):
        outcome["items"] = [
            {key: item[key] for key in ("title", "source", "url", "scope") if key in item}
            for item in details["items"][:5]
            if isinstance(item, dict)
        ]
    if details.get("metric_summary"):
        outcome["metric_summary"] = details["metric_summary"]
    return outcome


def _empty_journal() -> dict[str, Any]:
    return {
        "version": JOURNAL_VERSION,
        "last_action_id": 0,
        "action_counts": {},
        "milestones": [],
        "observations": {},
    }


def _apply_journal_action(journal: dict[str, Any], action_id: int, decision: dict[str, Any],
                          feedback: dict[str, Any]) -> None:
    action = decision.get("action", {})
    kind = action.get("kind", "invalid")
    journal["last_action_id"] = action_id
    counts = journal.setdefault("action_counts", {})
    counts[kind] = int(counts.get(kind, 0)) + 1
    key = observation_key(action)
    if key and feedback.get("status") == "ok":
        observations = journal.setdefault("observations", {})
        receipt = observations.get(key, {
            "observation_key": key,
            "first_action_id": action_id,
            "successful_reads": 0,
        })
        receipt.update({
            "last_action_id": action_id,
            "successful_reads": int(receipt.get("successful_reads", 0)) + 1,
            "action": _action_view(action),
            "evidence_ids": feedback.get("evidence_ids", [])[:10],
            "feedback_ref": f"action-{action_id}",
            "complete": not bool(feedback.get("details", {}).get("next")),
        })
        observations[key] = receipt
    milestone = (
        kind in PROGRESS_KINDS
        or kind == "plan"
        or kind.startswith("search_")
        or kind in ("fetch_source", "compare")
        or feedback.get("status") != "ok"
    )
    if milestone:
        journal.setdefault("milestones", []).append({
            "action_id": action_id,
            "decision_summary": str(decision.get("summary", ""))[:1000],
            "action": _action_view(action),
            "status": feedback.get("status"),
            "feedback_summary": str(feedback.get("summary", ""))[:500],
            "evidence_ids": feedback.get("evidence_ids", [])[:10],
            "feedback_ref": f"action-{action_id}",
            "outcome": _outcome_view(feedback),
            "advances_research": kind in PROGRESS_KINDS,
        })
        journal["milestones"] = journal["milestones"][-JOURNAL_MILESTONE_LIMIT:]


def rebuild_research_journal(store, *, persist: bool = True) -> dict[str, Any]:
    """Reconstruct the durable progress index solely from the authoritative ledger."""
    journal = _empty_journal()
    rows = store.db.execute(
        "SELECT id,decision,feedback FROM actions WHERE status='done' ORDER BY id"
    )
    for row in rows:
        decision = json.loads(row["decision"])
        feedback = json.loads(row["feedback"]) if row["feedback"] else {
            "status": "infrastructure_failure", "summary": "Missing persisted feedback",
        }
        _apply_journal_action(journal, int(row["id"]), decision, feedback)
    if persist:
        store.put("research_journal", journal)
    return journal


def ensure_research_journal(store) -> dict[str, Any]:
    journal = store.get("research_journal")
    row = store.db.execute("SELECT COALESCE(MAX(id),0) FROM actions WHERE status='done'").fetchone()
    last_done = int(row[0])
    if (not isinstance(journal, dict) or journal.get("version") != JOURNAL_VERSION
            or journal.get("last_action_id") != last_done):
        return rebuild_research_journal(store)
    return journal


def record_research_journal(store, action_id: int, decision: Any, feedback: Any) -> None:
    """Advance the journal in the same transaction that completes an action."""
    decision = decision.model_dump(mode="json") if hasattr(decision, "model_dump") else decision
    feedback = feedback.model_dump(mode="json") if hasattr(feedback, "model_dump") else feedback
    journal = store.get("research_journal")
    if not isinstance(journal, dict) or journal.get("version") != JOURNAL_VERSION:
        rebuild_research_journal(store)
        return
    if int(journal.get("last_action_id", 0)) != action_id - 1:
        rebuild_research_journal(store)
        return
    _apply_journal_action(journal, action_id, decision, feedback)
    store.put("research_journal", journal)


def _milestone_context_view(item: dict[str, Any]) -> dict[str, Any]:
    """Compact a rich persisted milestone without dropping its evidence handles."""
    action = dict(item.get("action", {}))
    if action.get("kind") == "plan" and isinstance(action.get("items"), list):
        action["items"] = [
            {**entry, "objective": str(entry.get("objective", ""))[:160]}
            for entry in action["items"][:8]
        ]
    outcome = dict(item.get("outcome", {}))
    if isinstance(outcome.get("items"), list):
        outcome["items"] = [
            {key: entry[key] for key in ("title", "source", "scope") if key in entry}
            for entry in outcome["items"][:3]
        ]
    return {
        "action_id": item.get("action_id"),
        "decision_summary": str(item.get("decision_summary", ""))[:500],
        "action": action,
        "status": item.get("status"),
        "feedback_summary": str(item.get("feedback_summary", ""))[:240],
        "evidence_ids": item.get("evidence_ids", [])[:6],
        "feedback_ref": item.get("feedback_ref"),
        "outcome": outcome,
        "advances_research": bool(item.get("advances_research")),
    }


def research_progress_view(store, milestone_limit: int = 16) -> dict[str, Any]:
    """Bounded resume state; full details remain addressable through evidence IDs."""
    journal = store.get("research_journal")
    if not isinstance(journal, dict) or journal.get("version") != JOURNAL_VERSION:
        journal = rebuild_research_journal(store, persist=False)
    observations = list(journal.get("observations", {}).values())
    observations.sort(key=lambda item: item.get("last_action_id", 0), reverse=True)
    duplicate_reads = sum(max(0, int(x.get("successful_reads", 0)) - 1) for x in observations)
    all_milestones = journal.get("milestones", [])
    progress = [item for item in all_milestones if item.get("advances_research")][-8:]
    latest_plan = next((item for item in reversed(all_milestones)
                        if item.get("action", {}).get("kind") == "plan"), None)
    evidence = [item for item in all_milestones if item.get("action", {}).get("kind") in {
        "search_literature", "search_web", "search_repositories", "fetch_source", "compare",
    }][-6:]
    failures = [item for item in all_milestones if item.get("status") != "ok"][-1:]
    selected = []
    selected_ids = set()
    for item in [*progress, *([latest_plan] if latest_plan else []), *evidence, *failures]:
        if item.get("action_id") not in selected_ids:
            selected.append(item)
            selected_ids.add(item.get("action_id"))
    selected.sort(key=lambda item: item.get("action_id", 0))
    while len(selected) > milestone_limit:
        removable = next((i for i, item in enumerate(selected) if not item.get("advances_research")), None)
        if removable is None:
            break
        selected.pop(removable)
    selected = [_milestone_context_view(item) for item in selected]
    return {
        "persistence": "SQLite fact ledger plus rebuildable structured journal",
        "journal_version": journal.get("version"),
        "last_recorded_action": journal.get("last_action_id", 0),
        "milestones": selected,
        "observation_inventory": {
            "unique": len(observations),
            "duplicate_successes": duplicate_reads,
        },
        "resume_instruction": (
            "Treat these milestones as completed work. Read old feedback with "
            "inspect(target='feedback', id='action-N') using its action ID; do not restart "
            "the investigation from scratch."
        ),
    }
