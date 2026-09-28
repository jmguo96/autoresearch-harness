"""Read only logs of known development trials, without granting arbitrary file access."""
from __future__ import annotations

import json
from pathlib import Path

from arlab.core.io import canonical, file_digest
from arlab.core.paging import text_page

STREAMS = {
    "train": "train/worker.log", "evaluation": "evaluation/worker.log",
    "metrics": "train/metrics.jsonl", "train_error": "train/error.json",
    "evaluation_error": "evaluation/error.json", "value_boundary": "train/value-boundary.json",
}
MAX_FILE_BYTES = 32 * 1024 * 1024


def read_trial_log(action, root: Path, contract, store) -> dict:
    row = store.trial(action.trial)
    dev = {c.name for c in contract.cases if c.split == "dev"}
    if row is None:
        raise ValueError("Unknown trial")
    if row["case"] not in dev:
        raise PermissionError("Only development trial diagnostics are available")
    if action.start_step is not None and action.end_step is not None and action.start_step > action.end_step:
        raise ValueError("Invalid step range")
    if action.stream != "metrics" and (action.start_step is not None or action.end_step is not None or action.fields):
        raise ValueError("Step/field filtering is only defined for metrics")
    base = root.resolve() / "jobs"
    path = base / action.trial / STREAMS[action.stream]
    if (not path.resolve().is_relative_to(base) or
            any(part.is_symlink() for part in [path, *path.parents] if part != base.parent)):
        raise PermissionError("Log path escapes the trial workspace or is symlinked")
    if not path.exists():
        return {"trial": action.trial, "stream": action.stream, "available": False,
                "reason": "The worker has not produced this diagnostic file", "continued_from": row.get("continued_from")}
    if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("Diagnostic file exceeds safe reading size; no unbounded read")
    # Read a bounded file, page arbitrary long lines without losing text.
    text = path.read_text(encoding="utf-8", errors="replace")
    rows = text.splitlines()
    original_count = len(rows)
    metric_summary = {}
    if action.stream == "metrics":
        chosen = []
        for line_number, line in enumerate(rows, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                chosen.append(canonical({"line": line_number, "parse_error": True, "preview": line[:200]}))
                continue
            step = record.get("step", record.get("tick"))
            if not isinstance(step, (int, float)):
                continue
            if (action.start_step is not None and step < action.start_step or
                    action.end_step is not None and step > action.end_step):
                continue
            for key, value in record.items():
                if isinstance(value, (int, float)) and key not in ("step", "tick"):
                    item = metric_summary.setdefault(key, {"first": value, "min": value, "max": value, "last": value})
                    item.update(min=min(item["min"], value), max=max(item["max"], value), last=value)
            chosen.append(canonical(record if not action.fields else
                                    {k: v for k, v in record.items() if k in action.fields or k in ("step", "tick")}))
        rows = chosen
    if action.tail:
        start_line = max(1, len(rows) - action.max_lines + 1)
    else:
        start_line = action.start_line
    page = text_page("\n".join(rows), start_line, action.max_lines, action.max_characters, action.offset)
    metadata = {"trial": action.trial, "stream": action.stream, "sha256": file_digest(path),
                "start_line": start_line, "scope": "development_diagnostics", "filtered": action.stream == "metrics"}
    evidence = store.save_artifact("log", canonical(page), metadata)
    return {"available": True, "trial": action.trial, "stream": action.stream, "evidence": evidence,
            "file_lines": original_count, "filtered_lines": len(rows),
            # Always retain the complete numeric summary. ``fields`` only controls
            # displayed rows, so a later request with another field list is not
            # needed merely to recover min/max/last statistics.
            "metric_summary": metric_summary,
            "continued_from": row.get("continued_from"), "training_reused_from": row.get("training_reused_from"), "note": "Only this segment's metrics; prior segments remain in their trial logs.", **page}
