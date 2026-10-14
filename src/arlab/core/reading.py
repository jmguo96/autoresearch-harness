"""Small, read-only tool handlers over persisted artifacts and experiment facts."""
from __future__ import annotations
import json
from arlab.core.io import digest
from arlab.core.paging import text_page, list_page
from arlab.core.panels import PanelIndex
from arlab.core.schema import Decision
from arlab.core.tool_catalog import TOOLS


def inspect_artifact(action, store, task, *, contract=None):
    from arlab.core.capabilities import decision_schema, enabled_tools
    available = enabled_tools(contract) if contract else TOOLS
    decision = decision_schema(contract) if contract else Decision
    if action.target == "skill":
        if action.id not in ("", task.name):
            raise ValueError("Unknown task skill")
        content = task.skill()
    elif action.target == "schema":
        if action.id in ("", "recipe", "expression", "algorithm"):
            content = task.recipe_schema()
        elif action.id == "decision":
            content = decision.model_json_schema()
        elif action.id in available:
            schema = decision.model_json_schema()
            name = schema["properties"]["action"]["discriminator"]["mapping"][action.id].rsplit("/", 1)[-1]
            content = {"schema": schema["$defs"][name],
                       "nested_definitions": "inspect(target='schema', id='decision')"}
        else:
            raise ValueError("Use inspect(target='schema', id='recipe' or 'decision'), or inspect(target='tool', id=<tool name>)")
    elif action.target == "tool":
        if action.id not in available:
            raise ValueError("Unknown tool")
        schema = decision.model_json_schema()
        mapping = schema["properties"]["action"]["discriminator"]["mapping"]
        name = mapping[action.id].rsplit("/", 1)[-1]
        content = {"description": available[action.id], "schema": schema["$defs"][name],
                   "nested_definitions": "inspect(schema, id='decision')"}
    elif action.target == "plan":
        content = store.get("plan", [])
    elif action.target == "feedback":
        reference = action.id or store.get("last_feedback_ref", "")
        if reference.startswith("action-") and reference[7:].isdigit():
            row = store.db.execute(
                "SELECT feedback FROM actions WHERE id=? AND status='done'",
                (int(reference[7:]),),
            ).fetchone()
            if row is None or row["feedback"] is None:
                raise ValueError("Unknown completed action feedback")
            content = row["feedback"]
        else:
            item = store.artifact(reference)
            if item is None or item["kind"] != "feedback":
                raise ValueError("Unknown tool-result artifact")
            content = item["text"]
    elif action.target == "trial":
        content = store.trial(action.id)
    elif action.target in ("candidate", "template"):
        candidate = store.candidate(action.id or store.get("focus_candidate") or store.get("baseline")) if action.target == "candidate" else None
        if action.target == "template" and not action.id:
            from arlab.algorithms.templates import TEMPLATES
            content = {"templates": list(TEMPLATES),
                       "next": "inspect(target='template', id=<template name>, path='candidate.py')"}
            return {"target": action.target, "id": action.id, "path": action.path,
                    **text_page(json.dumps(content, indent=2), action.start_line, action.max_lines, action.max_characters, action.offset)}
        recipe = candidate["recipe"] if candidate else (task.template(action.id) if action.target == "template" else None)
        if recipe is None:
            raise ValueError("Unknown candidate")
        if action.path:
            if recipe.get("kind") != "python" and task.name == "iql":
                recipe = task.template("iql-source")
            if action.path not in recipe.get("files", {}):
                raise ValueError(f"Unknown source path {action.path!r}; available files: {list(recipe.get('files', {}))}. Use inspect with no path to read the recipe.")
            content = recipe["files"][action.path]
        elif recipe.get("kind") == "python":
            content = {k: v for k, v in recipe.items() if k != "files"}
            content["files"] = [{"path": p, "lines": len(t.splitlines()), "characters": len(t)} for p, t in recipe["files"].items()]
            content["read_file"] = "inspect with path and start_line; exact replace editing uses raw source text"
        else:
            content = recipe
    else:
        raise ValueError("Unsupported inspection target")
    if content is None:
        raise ValueError("Artifact does not exist")
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, indent=2)
    return {"target": action.target, "id": action.id, "path": action.path,
            **({"source_origin": "iql-source", "edit_hint": "edit_algorithm with this expression parent automatically creates an editable IQL source copy"}
               if action.target == "candidate" and action.path and candidate["recipe"].get("kind") != "python" and task.name == "iql" else {}),
            **text_page(text, action.start_line, action.max_lines, action.max_characters, action.offset)}


def query_experiments(action, contract, store):
    from arlab.core.branches import branch_state
    branches = branch_state(store)
    candidates, records = store.candidates(), store.trials()
    ids = [c["id"] for c in candidates]
    if action.candidate and action.candidate not in ids:
        raise ValueError("Unknown candidate")
    if action.case and action.target not in ("trials", "checks", "missing"):
        raise ValueError("case filtering is supported for trials, checks and missing cells, not aggregated panels")
    if action.status and action.target not in ("trials", "checks", "missing"):
        raise ValueError("status filtering is supported for trials, checks and missing cells")
    dev = {c.name for c in contract.cases if c.split == "dev"}
    if action.case and action.case not in dev:
        raise PermissionError("Only development experiment queries are authorized")
    index = PanelIndex(contract, store.panel_trials())
    if action.target in ("trials", "checks"):
        check = action.target == "checks"
        legacy = set((store.get("source_migration") or {}).get("legacy_trials", []))
        rows = [{**{k: t[k] for k in ("trial_id", "candidate", "case", "train_seed", "steps", "status", "score", "metric", "purpose", "eval_seeds", "continued_from", "incremental_train_steps", "source_sha256") if k in t},
                 "legacy_source": t.get("trial_id") in legacy}
                for t in records if (t.get("purpose") == "check") == check and t.get("case") in dev
                and (not action.candidate or t["candidate"] == action.candidate)
                and (not action.case or t["case"] == action.case)
                and (not action.steps or t["steps"] == action.steps)
                and (not action.status or t["status"] == action.status)]
    elif action.target == "candidates":
        rows = [{"id": c["id"], "parent": c["parent"], "name": c["recipe"].get("name"),
                 "kind": c["recipe"].get("kind", "expression"), "recipe_hash": digest(c["recipe"]),
                 "branch": branches[c["id"]]["branch"], "active": branches[c["id"]]["active"],
                 "question": c["hypothesis"].get("question", "")[:400]}
                for c in candidates if not action.candidate or c["id"] == action.candidate]
    elif action.target in ("missing", "panels"):
        if action.steps and not index.legal_steps(action.steps):
            raise ValueError("Unknown experimental rung")
        rows = []
        for cid in ([action.candidate] if action.candidate else ids):
            for step in ([action.steps] if action.steps else contract.rungs):
                if action.target == "missing":
                    rows.extend(index.missing(cid, step))
                else:
                    rows.append(index.panel(cid, step))
        if action.case:
            rows = [r for r in rows if r.get("case") == action.case]
        if action.status:
            rows = [r for r in rows if r.get("status") == action.status]
    elif action.target == "sources":
        total = store.db.execute("SELECT COUNT(*) FROM artifacts WHERE kind='src'").fetchone()[0]
        items = [{"id": x["id"], "title": x["metadata"].get("title"), "scope": x["metadata"].get("scope"),
                  "url": x["metadata"].get("url"), "characters": x["characters"]}
                 for x in store.artifact_index("src", action.limit, action.offset)]
        return {"items": items, "total": total, "next_offset": action.offset + len(items) if action.offset + len(items) < total else None}
    else:
        raise ValueError("Unknown query target")
    return {"target": action.target, **list_page(rows, action.offset, action.limit)}
