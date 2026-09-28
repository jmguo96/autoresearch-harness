"""Immutable algorithm revisions and bounded active hypotheses are different things.

All versions stay addressable. Archiving a branch removes it from active planning,
not from history or final selection. The baseline can never be archived.
"""
from __future__ import annotations

from arlab.core.io import digest


def branch_state(store) -> dict:
    state = store.get("branches", {})
    # Support fixtures / callers registering candidates directly in Store.
    for candidate in store.candidates():
        cid = candidate["id"]
        if cid not in state:
            state[cid] = {"branch": cid, "active": True}
    return state


def register_revision(store, contract, recipe, hypothesis, parent: str, *, new_branch: bool) -> str:
    cid = "c-" + digest(recipe)[:20]
    if store.candidate(cid):
        return cid
    if len(store.candidates()) >= contract.max_revisions:
        raise PermissionError("Revision storage budget reached; historical versions are not deleted")
    state = branch_state(store)
    baseline = store.get("baseline")
    create = new_branch or parent == baseline
    branch = cid if create else state[parent]["branch"]
    active_branches = {r["branch"] for r in state.values() if r["active"]}
    if (create or branch not in active_branches) and len(active_branches) >= contract.max_candidates:
        raise PermissionError("Active branch limit reached; archive a direction or revise an active one")
    with store.transaction():
        store.add_candidate(recipe, hypothesis, parent)
        for record in state.values():
            if record["branch"] == branch:
                record["active"] = True
        state[cid] = {"branch": branch, "active": True}
        store.put("branches", state)
    return cid


def set_branch_status(store, contract, candidate: str, active: bool) -> str:
    if not store.candidate(candidate):
        raise ValueError("Unknown candidate")
    state = branch_state(store)
    branch = state[candidate]["branch"]
    baseline = store.get("baseline")
    if branch == state[baseline]["branch"] and not active:
        raise PermissionError("Fixed baseline cannot be archived")
    active_branches = {r["branch"] for r in state.values() if r["active"]}
    if active and branch not in active_branches and len(active_branches) >= contract.max_candidates:
        raise PermissionError("Active branch limit reached")
    for record in state.values():
        if record["branch"] == branch:
            record["active"] = active
    store.put("branches", state)
    return branch
