"""v0.6: real tiny training for continuation, explicit fixtures for control/claims.

Nothing here is a D4RL performance result. No live model or research API is used.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from arlab.algorithms.runtime import train_code
from arlab.algorithms.templates import template
from arlab.audit import audit_campaign
from arlab.classroom.reporting import study_report
from arlab.core.branches import branch_state
from arlab.core.checkpoints import series_binding, training_binding
from arlab.core.diagnostics import read_trial_log
from arlab.core.experiments import TrialService
from arlab.core.harness import Harness
from arlab.core.io import file_digest, read_json, write_json
from arlab.core.panels import PanelIndex
from arlab.core.schema import (
    BranchStatusAction, Case, CompareAction, ContinueTrialAction, Decision,
    ExplorationPolicy, FinalEvaluation, FinishAction, Hypothesis, ProposeAction,
    ReadTrialLogAction, ResearchContract, RunAction,
)
from arlab.core.store import BudgetExceeded
from arlab.tasks.iql.environments import make_fixture
from arlab.tasks.iql.recipe import IQLRecipe
from arlab.tasks.iql.trainer import train
from arlab.cli import demo_contract


def adaptive_contract(tmp_path, *, horizon=12, tasks=3, final=False, open_code=False):
    raw = demo_contract(tmp_path / "data", steps=horizon, candidates=4, open_code=open_code).model_dump(mode="json")
    raw.update(rungs=sorted({min(4, horizon), horizon}), train_seeds=[0, 1, 2, 3],
               exploration=ExplorationPolicy(adaptive=True, minimum_steps=1, initial_steps=4,
                   initial_eval_episodes=1, confirmation_seeds=[0, 1, 2]).model_dump(mode="json"))
    raw["cases"] = []
    for i in range(tasks):
        path = tmp_path / f"data-{i}.npz"
        sha = make_fixture(path, dynamics_seed=i, episodes=4, horizon=4)
        raw["cases"].append(Case(name=f"fixture-{i}", data=str(path), dataset_sha256=sha,
            environment={"kind": "synthetic_lq", "dynamics_seed": i, "horizon": 4}).model_dump(mode="json"))
    raw["budget"].update(max_trials=200, max_attempts=400, max_checks=50, max_train_steps=100000,
                         max_optimizer_steps=1000000, max_eval_episodes=2000, max_decisions=1000)
    if final:
        raw["final_evaluation"] = FinalEvaluation(cases=[c["name"] for c in raw["cases"]],
            steps=horizon, train_seeds=[41], eval_seeds=[9000]).model_dump(mode="json")
    return ResearchContract.model_validate(raw)


def hypothesis():
    return Hypothesis(question="Test fixture", prediction="Unknown", rationale="Protocol test",
                      falsification="Failure; not benchmark evidence")


def tensor_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            tensor_tree_equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            tensor_tree_equal(x, y)
    else:
        assert a == b


def request(dataset, *, code=None, steps=5, horizon=18):
    return {"recipe": template(code, tiny=True) if code else IQLRecipe(hidden_dims=[16], batch_size=16).model_dump(mode="json"),
            "data": str(dataset[0]), "dataset_sha256": dataset[1], "seed": 7,
            "steps": steps, "schedule_steps": horizon, "device": "cpu",
            "code_execution": {"mode": "trusted_local"} if code else {"mode": "disabled"}}


def child_request(raw, parent, trained, target):
    child = copy.deepcopy(raw)
    child["steps"] = target
    child["resume_from"] = {"path": str(parent / "checkpoint.pt"),
        "sha256": trained["checkpoint_sha256"], "series_sha256": trained["series_sha256"],
        "request_sha256": trained["request_sha256"], "steps": trained["steps"]}
    return child


@pytest.mark.parametrize("code", [None, "ensemble-mixture", "diffusion-extraction"])
def test_segmented_training_equals_uninterrupted_including_optimizer_and_rng(dataset, tmp_path, code):
    fn = train_code if code else train
    raw = request(dataset, code=code)
    first = fn(raw, tmp_path / "first")
    middle_req = child_request(raw, tmp_path / "first", first, 11)
    middle = fn(middle_req, tmp_path / "middle")
    last_req = child_request(raw, tmp_path / "middle", middle, 18)
    last = fn(last_req, tmp_path / "last")
    uninterrupted = {**raw, "steps": 18}
    fn(uninterrupted, tmp_path / "full")
    assert first["resumed_at"] == 0 and middle["resumed_at"] == 5 and last["resumed_at"] == 11
    a = torch.load(tmp_path / "last/checkpoint.pt", weights_only=True)
    b = torch.load(tmp_path / "full/checkpoint.pt", weights_only=True)
    for k in ("algorithm" if code else "trainer", "torch_rng", "replay_rng", "cuda_rng"):
        tensor_tree_equal(a[k], b[k])
    if code:
        for k in ("optimizer_steps", "python_rng", "numpy_rng"):
            tensor_tree_equal(a[k], b[k])
    x = load_file(str(tmp_path / "last/artifact/weights.safetensors"))
    y = load_file(str(tmp_path / "full/artifact/weights.safetensors"))
    tensor_tree_equal(x, y)
    # The earlier endpoint is untouched by continuation.
    assert file_digest(tmp_path / "first/checkpoint.pt") == first["checkpoint_sha256"]


@pytest.mark.parametrize("field", ["seed", "schedule_steps", "recipe", "hash", "step_receipt"])
def test_continuation_rejects_incompatible_or_tampered_source(dataset, tmp_path, field):
    raw = request(dataset)
    first = train(raw, tmp_path / "first")
    child = child_request(raw, tmp_path / "first", first, 11)
    if field == "recipe":
        child["recipe"]["beta"] = 0.9
    elif field == "hash":
        child["resume_from"]["sha256"] = "0" * 64
    elif field == "step_receipt":
        child["resume_from"]["steps"] = 4
    else:
        child[field] += 1
    with pytest.raises(ValueError):
        train(child, tmp_path / "child")


def test_revisions_do_not_exhaust_active_hypothesis_budget(tmp_path):
    raw = adaptive_contract(tmp_path).model_dump(mode="json")
    raw.update(max_candidates=2, max_revisions=200)
    h = Harness(ResearchContract.model_validate(raw), tmp_path / "run")
    try:
        h._initialize()
        parent = h.store.get("baseline")
        for i in range(100):
            recipe = copy.deepcopy(h.contract.seed_recipe)
            recipe["name"] = f"reviewed-revision-{i}"
            result = h._execute(Decision(summary="Revision fixture", action=ProposeAction(kind="propose",
                parent=parent, recipe=recipe, hypothesis=hypothesis(), new_branch=(i == 0))))
            parent = result.details["candidate"]
        assert len(h.store.candidates()) == 101
        assert len({r["branch"] for r in branch_state(h.store).values()}) == 2
        assert h._context()["branch_budget"]["active"] == 2
        h._execute(Decision(summary="Archive completed direction", action=BranchStatusAction(
            kind="set_branch_status", candidate=parent, active=False, reason="Try a different hypothesis")))
        recipe["name"] = "new-independent-direction"
        h._execute(Decision(summary="New branch", action=ProposeAction(kind="propose", parent=h.store.get("baseline"),
            recipe=recipe, hypothesis=hypothesis())))
        assert len(h.store.candidates()) == 102
    finally:
        h.close()


def test_baseline_cannot_be_archived(tmp_path):
    h = Harness(adaptive_contract(tmp_path), tmp_path / "run")
    try:
        h._initialize()
        with pytest.raises(PermissionError, match="baseline"):
            h._execute(Decision(summary="Invalid fixture", action=BranchStatusAction(kind="set_branch_status",
                candidate=h.store.get("baseline"), active=False, reason="Not allowed")))
    finally:
        h.close()


@pytest.mark.integration
def test_trial_service_continues_only_delta_and_can_expand_rollouts_without_training(tmp_path):
    h = Harness(adaptive_contract(tmp_path, tasks=2), tmp_path / "run")
    try:
        h._initialize()
        cid = h.store.get("baseline")
        a = h.service.run(RunAction(kind="run", candidate=cid, case="fixture-0", train_seed=0, steps=4, eval_episodes=1))
        b = h.service.continue_trial(ContinueTrialAction(kind="continue_trial", trial=a["trial_id"], steps=9, eval_episodes=1))
        assert b["resumed_at"] == 4 and b["incremental_train_steps"] == 5
        assert h.store.totals()["train_steps"] == 9
        c = h.service.continue_trial(ContinueTrialAction(kind="continue_trial", trial=b["trial_id"], steps=9, eval_episodes=3))
        assert c["training_reused_from"] == b["trial_id"] and c["incremental_train_steps"] == 0
        assert h.store.totals()["train_steps"] == 9
        before = h.store.totals()
        assert h.service.run(RunAction(kind="run", candidate=cid, case="fixture-0", train_seed=0, steps=9, eval_episodes=3))["trial_id"] == c["trial_id"]
        assert h.store.totals() == before
        # New seed is genuinely independent; it does not inherit the old checkpoint.
        other = h.service.run(RunAction(kind="run", candidate=cid, case="fixture-0", train_seed=1, steps=4, eval_episodes=1))
        assert other["resumed_at"] == 0 and h.store.totals()["train_steps"] == 13
        env = h.service.run(RunAction(kind="run", candidate=cid, case="fixture-1", train_seed=0, steps=4, eval_episodes=1))
        assert env["resumed_at"] == 0 and h.store.totals()["train_steps"] == 17
        detail = read_trial_log(ReadTrialLogAction(kind="read_trial_log", trial=b["trial_id"], stream="metrics",
            start_step=5, end_step=9, fields=["critic_loss"]), h.root, h.contract, h.store)
        assert detail["available"] and detail["metric_summary"]
        assert detail["continued_from"] == a["trial_id"]
    finally:
        h.close()


@pytest.mark.integration
def test_open_source_continuation_also_flows_through_independent_evaluation(tmp_path):
    h = Harness(adaptive_contract(tmp_path, tasks=1, open_code=True), tmp_path / "run")
    try:
        h._initialize()
        cid = h.store.add_candidate(template("ensemble-mixture", tiny=True), {"question": "integration"}, h.store.get("baseline"))
        a = h.service.run(RunAction(kind="run", candidate=cid, case="fixture-0", train_seed=0, steps=3, eval_episodes=1))
        b = h.service.continue_trial(ContinueTrialAction(kind="continue_trial", trial=a["trial_id"], steps=6, eval_episodes=1))
        assert b["resumed_at"] == 3 and b["incremental_train_steps"] == 3
        assert b["schedule_steps"] == 12 and b["returns"]
    finally:
        h.close()


def fixture_row(h, cid, case, seed, steps, *, episodes=1, score=1.0):
    tid = f"fixture-{cid}-{case}-{seed}-{steps}-{episodes}"
    row = {"trial_id": tid, "candidate": cid, "case": case, "train_seed": seed, "steps": steps,
           "status": "complete", "score": score, "metric": "synthetic_fixture_NOT_D4RL",
           "eval_seeds": h.contract.eval_seeds[:episodes], "schedule_steps": h.contract.rungs[-1]}
    h.store.save_trial(tid, cid, "complete", row)
    return row


def test_explicit_partial_compare_informs_exploration_but_cannot_finish(tmp_path):
    h = Harness(adaptive_contract(tmp_path), tmp_path / "run")
    try:
        h._initialize()
        baseline = h.store.get("baseline")
        recipe = {**h.contract.seed_recipe, "name": "subset-fixture"}
        cid = h.store.add_candidate(recipe, {}, baseline)
        for x in [baseline, cid]:
            fixture_row(h, x, "fixture-0", 0, 7)
        result = PanelIndex(h.contract, h.store.trials()).compare(baseline, cid, 7,
            cases=["fixture-0"], seeds=[0], episodes=1)
        assert result["selected_panel_complete"] and not result["contract_panel_complete"]
        assert result["pairs"] == 1 and not result["freeze_eligible"]
        assert "exploratory_subset" in result["scope"]
        with pytest.raises(ValueError, match="panel"):
            h._execute(Decision(summary="Illegal finish", action=FinishAction(kind="finish", candidate=cid, conclusion="fixture")))
    finally:
        h.close()


def test_different_rollout_counts_are_not_mixed_and_extra_seed_is_not_required(tmp_path):
    h = Harness(adaptive_contract(tmp_path), tmp_path / "run")
    try:
        h._initialize()
        base = h.store.get("baseline")
        for case in [f"fixture-{i}" for i in range(3)]:
            for seed in [0, 1, 2]:
                fixture_row(h, base, case, seed, 12, episodes=1)
                fixture_row(h, base, case, seed, 12, episodes=3)
        panel = PanelIndex(h.contract, h.store.trials()).panel(base, 12)
        assert panel["complete"] and panel["completed"] == 9  # Not 18, and not 12 seeds/cells.
        h._execute(Decision(summary="Complete baseline panel", action=FinishAction(kind="finish", candidate=base, conclusion="No improvement claim")))
        assert h.store.get("status") == "completed"
    finally:
        h.close()


@pytest.mark.parametrize("missing", ["environment", "seed", "episodes"])
def test_final_development_gate_still_requires_full_three_distribution_panel(tmp_path, missing):
    h = Harness(adaptive_contract(tmp_path), tmp_path / "run")
    try:
        h._initialize()
        base = h.store.get("baseline")
        for i in range(3):
            for seed in range(3):
                if missing == "environment" and i == 2 or missing == "seed" and seed == 2:
                    continue
                fixture_row(h, base, f"fixture-{i}", seed, 12, episodes=1 if missing == "episodes" else 3)
        assert not h._ranking(final=True)
    finally:
        h.close()


def test_context_exposes_remaining_decisions_and_wall_time(tmp_path):
    h = Harness(adaptive_contract(tmp_path), tmp_path / "run")
    try:
        h._initialize()
        before = h._context()["termination_budget"]
        for _ in range(3):
            h.store.begin_action(Decision(summary="fixture", action=RunAction(kind="run", candidate=h.store.get("baseline"), case="fixture-0", train_seed=0, steps=4)))
        h.store.put("started_at", time.time() - h.contract.budget.max_wall_seconds + 1)
        after = h._context()["termination_budget"]
        assert after["decisions_remaining"] == before["decisions_remaining"] - 3
        assert 0 <= after["wall_seconds_remaining"] <= 1.1
        assert after["pending_action_recoverable"]
    finally:
        h.close()


@pytest.mark.integration
def test_failure_traceback_is_available_to_researcher(tmp_path):
    h = Harness(adaptive_contract(tmp_path, tasks=1, open_code=True), tmp_path / "run")
    try:
        h._initialize()
        recipe = template("ensemble-mixture", tiny=True)
        recipe["files"]["candidate.py"] += "\n\ndef build(spec):\n    return spec.config['missing_review_fixture_key']\n"
        cid = h.store.add_candidate(recipe, {}, h.store.get("baseline"))
        with pytest.raises(RuntimeError, match="missing_review_fixture_key"):
            h.service.run(RunAction(kind="run", candidate=cid, case="fixture-0", train_seed=0, steps=2), check=True)
        tid = h.store.get("last_failed_trial")
        page = read_trial_log(ReadTrialLogAction(kind="read_trial_log", trial=tid, stream="train_error"), h.root, h.contract, h.store)
        assert "KeyError" in str(page) and "candidate.py" in str(page)
        assert page["evidence"] and h.store.evidence_exists(page["evidence"])
    finally:
        h.close()


def test_log_reader_refuses_symlink_and_final_task(tmp_path):
    h = Harness(adaptive_contract(tmp_path), tmp_path / "run")
    try:
        h._initialize()
        base = h.store.get("baseline")
        row = fixture_row(h, base, "fixture-0", 0, 4)
        path = h.root / "jobs" / row["trial_id"] / "train"
        path.mkdir(parents=True)
        (path / "worker.log").symlink_to(tmp_path / "secret")
        with pytest.raises(PermissionError):
            read_trial_log(ReadTrialLogAction(kind="read_trial_log", trial=row["trial_id"]), h.root, h.contract, h.store)
        h.store.save_trial("test-only", base, "failed", {**row, "trial_id": "test-only", "case": "reserved-final"})
        with pytest.raises(PermissionError):
            read_trial_log(ReadTrialLogAction(kind="read_trial_log", trial="test-only"), h.root, h.contract, h.store)
    finally:
        h.close()


def freeze_baseline(h):
    base = h.store.get("baseline")
    for c in h.contract.cases:
        if c.split == "dev":
            for s in h.contract.exploration.confirmation_seeds or h.contract.train_seeds:
                fixture_row(h, base, c.name, s, h.contract.rungs[-1], episodes=len(h.contract.eval_seeds))
    h._execute(Decision(summary="Fixture freeze", action=FinishAction(kind="finish", candidate=base, conclusion="Test fixture, not measured development results")))
    h.report()


@pytest.mark.integration
def test_audit_recovers_from_pretraining_failure_with_authorized_headroom(tmp_path, monkeypatch):
    h = Harness(adaptive_contract(tmp_path, horizon=4, tasks=3, final=True), tmp_path / "development")
    try:
        h._initialize()
        freeze_baseline(h)
    finally:
        h.close()
    original = TrialService._worker
    calls = []
    def once(self, mode, request, output, read_only):
        calls.append(mode)
        if len(calls) == 1:
            raise RuntimeError("Injected transient failure before training, NOT a score")
        return original(self, mode, request, output, read_only)
    monkeypatch.setattr(TrialService, "_worker", once)
    result = audit_campaign(tmp_path / "development", tmp_path / "final", split="final")
    assert result["status"] == "completed" and len(result["trials"]) == 3
    assert result["reserved"]["train_steps"] == 16  # 12 useful + 4 conservatively retained.
    assert result["actual_known_usage"]["train_steps"] == 12
    assert result["protocol"]["budget_estimate"]["recovery_capacity"]["train_steps"] >= 16


def test_summary_does_not_call_development_only_gain_independent_improvement(tmp_path):
    # Nine fixture tasks; only the three development tasks improve strongly.
    raw = adaptive_contract(tmp_path, tasks=9, final=True).model_dump(mode="json")
    for c in raw["cases"][3:]:
        c["split"] = "test"
    raw["final_evaluation"]["train_seeds"] = [41, 43, 47]
    contract = ResearchContract.model_validate(raw)
    trials = []
    for c in contract.cases:
        for seed in contract.final_evaluation.train_seeds:
            for cid in ["base", "new"]:
                score = 0.0 if cid == "base" else (100.0 if c.split == "dev" else -1.0)
                trials.append({"candidate": cid, "case": c.name, "train_seed": seed, "steps": 12,
                               "metric": "fixture_NOT_D4RL", "score": score})
    write_json(tmp_path / "final/audit.json", {"status": "completed", "baseline_candidate": "base",
        "selected_candidate": "new", "trials": trials})
    report = study_report(tmp_path, contract)
    assert report["research_complete"] and report["all_tasks_descriptive_gain_positive"]
    assert not report["method_improved"] and not report["not_used_for_search_improvement_supported"]
    assert report["not_used_for_search_evidence"]["macro_delta"] == -1.0


def test_high_default_budget_and_dynamic_capabilities(tmp_path):
    from test_classroom import nine_contract
    from arlab.core.capabilities import enabled_tools
    contract = nine_contract(tmp_path)
    assert contract.max_candidates == 128 and contract.max_revisions == 10000
    assert contract.budget.max_trials == 20000 and contract.budget.max_decisions == 20000
    assert contract.exploration.initial_steps == 5000
    assert contract.exploration.confirmation_seeds == [0, 1, 2]
    assert contract.train_seeds == list(range(10))
    assert "continue_trial" in enabled_tools(contract)
    assert "read_trial_log" in enabled_tools(contract)
    assert "search_web" not in enabled_tools(contract)
    assert "continue_trial" not in enabled_tools(demo_contract(tmp_path / "plain", steps=2))


def test_mismatched_rollout_counts_never_make_a_pair(tmp_path):
    h = Harness(adaptive_contract(tmp_path), tmp_path / "run")
    try:
        h._initialize()
        base = h.store.get("baseline")
        cid = h.store.add_candidate({**h.contract.seed_recipe, "name": "unequal-rollout-fixture"}, {}, base)
        fixture_row(h, base, "fixture-0", 0, 4, episodes=1)
        fixture_row(h, cid, "fixture-0", 0, 4, episodes=3)
        result = PanelIndex(h.contract, h.store.trials()).compare(base, cid, 4,
            cases=["fixture-0"], seeds=[0], episodes=3)
        assert result["pairs"] == 0 and not result["selected_panel_complete"]
    finally:
        h.close()


def test_attempt_cap_persists_across_recovery_and_unknown_work_not_refunded(tmp_path, monkeypatch):
    h = Harness(adaptive_contract(tmp_path, tasks=1), tmp_path / "run")
    try:
        h._initialize()
        calls = []
        def fail(*args):
            calls.append(1)
            raise RuntimeError("Injected failure, not benchmark data")
        monkeypatch.setattr(h.service, "_worker", fail)
        a = RunAction(kind="run", candidate=h.store.get("baseline"), case="fixture-0", train_seed=0, steps=4)
        for _ in range(3):
            with pytest.raises(RuntimeError, match="Injected"):
                h.service.run(a)
        before = h.store.totals()
        with pytest.raises(BudgetExceeded, match="retry limit"):
            h.service.run(a)
        assert h.store.totals() == before and before["train_steps"] == 12
        assert len(calls) == 3
    finally:
        h.close()


def test_revision_hard_cap_is_still_enforced(tmp_path):
    raw = adaptive_contract(tmp_path).model_dump(mode="json")
    raw["max_revisions"] = 2
    h = Harness(ResearchContract.model_validate(raw), tmp_path / "run")
    try:
        h._initialize()
        base = h.store.get("baseline")
        a = ProposeAction(kind="propose", parent=base, recipe={**raw["seed_recipe"], "name": "one"}, hypothesis=hypothesis())
        first = h._execute(Decision(summary="fixture", action=a)).details["candidate"]
        with pytest.raises(PermissionError, match="Revision storage"):
            h._execute(Decision(summary="fixture", action=ProposeAction(kind="propose", parent=first,
                new_branch=False, recipe={**raw["seed_recipe"], "name": "two"}, hypothesis=hypothesis())))
    finally:
        h.close()
