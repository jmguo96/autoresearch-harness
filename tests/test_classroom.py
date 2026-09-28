"""Student UX and full research protocol regressions.

Synthetic ledgers below are explicitly protocol fixtures, NEVER D4RL scores.
The integration test at the end executes real tiny IQL runs on synthetic tasks.
"""
from __future__ import annotations

import argparse
import copy
import io
import json
from pathlib import Path

import pytest

from arlab.audit import audit_campaign
from arlab.classroom.bootstrap import installation_command, parser as entry_parser
from arlab.classroom.constants import D4RL_SUITE9, EXPLORATION_TASKS
from arlab.classroom.reporting import study_report
from arlab.classroom.session import budget_preview, model_settings, require_key, validate_endpoint
from arlab.cli import demo_contract
from arlab.core.capabilities import authorize_action, decision_schema, enabled_tools
from arlab.core.harness import Harness
from arlab.core.io import digest, file_digest, read_json, write_json
from arlab.core.schema import (
    Case, Decision, FinalEvaluation, FinishAction, ModelConfig, ResearchContract, RunAction,
)
from arlab.core.store import Store
from arlab.tasks.iql.environments import make_fixture
from arlab.tasks.iql.study import apply_nine_task_protocol, require_nine_task_protocol


def nine_contract(tmp_path):
    """Real nine-task identities, fabricated hashes, no downloaded D4RL data."""
    raw = demo_contract(tmp_path / "seed", steps=2).model_dump(mode="json")
    raw["cases"] = [{"name": name, "split": "dev", "data": str(tmp_path / f"{name}.npz"),
                     "environment": {"kind": "d4rl", "id": name}, "dataset_sha256": "0" * 64}
                    for name in D4RL_SUITE9]
    return apply_nine_task_protocol(raw)


def protocol_fixture(tmp_path, cases=3, steps=2, seeds=(41, 43), dev_count=1):
    """Synthetic worlds with a real contract, used for execution or controlled ledgers."""
    raw = demo_contract(tmp_path / "data-root", steps=steps, candidates=1).model_dump(mode="json")
    declarations = []
    for index in range(cases):
        path = tmp_path / f"fixture-{index}.npz"
        make_fixture(path, dynamics_seed=index, data_seed=31, episodes=6, horizon=4)
        declarations.append(Case(name=f"fixture-{index}", split="dev" if index < dev_count else "test",
            data=str(path), dataset_sha256=file_digest(path),
            environment={"kind": "synthetic_lq", "dynamics_seed": index, "horizon": 4}).model_dump(mode="json"))
    raw["cases"] = declarations
    raw["final_evaluation"] = FinalEvaluation(cases=[c["name"] for c in declarations],
        train_seeds=list(seeds), eval_seeds=[9000], steps=steps).model_dump(mode="json")
    return ResearchContract.model_validate(raw)


def freeze_fixture(contract, campaign, distinct=True):
    """Control-flow fixture, not measured performance."""
    agent = Harness(contract, campaign)
    try:
        agent._initialize()
        baseline = agent.store.get("baseline")
        if distinct:
            recipe = copy.deepcopy(contract.seed_recipe)
            recipe["name"] = "protocol-fixture-candidate"
            candidate = agent.store.add_candidate(recipe, {"question": "protocol fixture"}, baseline)
        else:
            candidate = baseline
        for cid in dict.fromkeys([baseline, candidate]):
            for case in [c for c in contract.cases if c.split == "dev"]:
                for seed in contract.train_seeds:
                    tid = f"fixture-{cid}-{case.name}-{seed}"
                    agent.store.save_trial(tid, cid, "complete", {
                        "trial_id": tid, "candidate": cid, "case": case.name, "train_seed": seed,
                        "steps": contract.rungs[-1], "score": 0.0, "metric": "synthetic_fixture",
                        "purpose": "experiment", "evidence_type": "test_fixture_NOT_training"})
        agent._execute(Decision(summary="protocol fixture freeze", action=FinishAction(
            kind="finish", candidate=candidate, conclusion="Protocol fixture, not scientific evidence.")))
        agent.report()
        return baseline, candidate
    finally:
        agent.close()


class StubTrials:
    calls = []
    fail_on = None

    def __init__(self, root, contract, store, task):
        self.store = store

    def run(self, action, **kwargs):
        tid = f"stub-{action.candidate}-{action.case}-{action.train_seed}"
        cached = self.store.trial(tid)
        if cached and cached["status"] == "complete":
            return cached
        self.calls.append((action, kwargs))
        if self.fail_on == action.case:
            raise RuntimeError("Controlled fixture failure, NOT an algorithm score")
        row = {"trial_id": tid, "candidate": action.candidate, "case": action.case,
               "train_seed": action.train_seed, "steps": action.steps, "score": 1.0,
               "metric": "synthetic_fixture", "evidence_type": "test_fixture_NOT_training"}
        self.store.save_trial(tid, action.candidate, "complete", row)
        return row


@pytest.fixture
def stub(monkeypatch):
    StubTrials.calls = []
    StubTrials.fail_on = None
    monkeypatch.setattr("arlab.audit.TrialService", StubTrials)
    return StubTrials


def test_default_entry_full_nine_tasks_not_small_budget(tmp_path):
    args = entry_parser().parse_args([])
    command = installation_command(tmp_path, tmp_path / ".arlab", args)
    assert "--suite9" in command and "--suite12" not in command
    assert "--research-online" in command and args.code_mode == "trusted_local"
    with pytest.raises(SystemExit):
        entry_parser().parse_args(["--full"])


def test_nine_task_preset_complete_protocol(tmp_path):
    contract = nine_contract(tmp_path)
    require_nine_task_protocol(contract)
    assert len(contract.cases) == 9
    assert not any("random" in c.name for c in contract.cases)
    assert [c.name for c in contract.cases if c.split == "dev"] == list(EXPLORATION_TASKS)
    assert len([c for c in contract.cases if c.split == "test"]) == 6
    assert set(contract.final_evaluation.cases) == set(D4RL_SUITE9)
    assert contract.final_evaluation.steps == 1_000_000
    assert len(contract.final_evaluation.train_seeds) == 5
    preview = budget_preview(contract)
    assert preview["final_runs_if_candidate_differs"] == 90
    assert preview["final_ticks_if_candidate_differs"] == 90_000_000
    assert preview["final_runs_if_IQL_retained"] == 45
    assert contract.research_access.enabled
    assert not contract.research_access.web_search_enabled


def test_final_protocol_rejects_missing_case_and_seed_reuse(tmp_path):
    contract = nine_contract(tmp_path)
    raw = contract.model_dump(mode="json")
    raw["final_evaluation"]["cases"] = ["made-up"]
    with pytest.raises(ValueError, match="undeclared"):
        ResearchContract.model_validate(raw)
    raw = contract.model_dump(mode="json")
    raw["final_evaluation"]["train_seeds"] = [0, 1, 2]
    with pytest.raises(ValueError, match="fresh"):
        ResearchContract.model_validate(raw)


@pytest.mark.parametrize("field,value", [("steps", 100), ("train_seeds", [41]), ("eval_seeds", [9000])])
def test_student_protocol_cannot_shrink_final_evidence(tmp_path, field, value):
    raw = nine_contract(tmp_path).model_dump(mode="json")
    raw["final_evaluation"][field] = value
    with pytest.raises(ValueError):
        require_nine_task_protocol(ResearchContract.model_validate(raw))


def test_web_search_removed_from_menu_schema_and_executor(tmp_path):
    contract = demo_contract(tmp_path, steps=2)
    contract.research_access.enabled = True
    names = enabled_tools(contract)
    assert "search_literature" in names and "search_repositories" in names
    assert "search_web" not in names
    schema = decision_schema(contract)
    assert "SearchWebAction" not in schema.model_json_schema().get("$defs", {})
    with pytest.raises(ValueError):
        schema.model_validate({"summary": "not allowed", "action": {"kind": "search_web", "query": "IQL"}})
    with pytest.raises(PermissionError):
        authorize_action(contract, "search_web")
    allowed = schema.model_validate({"summary": "read", "action": {"kind": "search_literature", "query": "IQL"}})
    assert allowed.action.kind == "search_literature"


def test_all_network_opt_out_keeps_reading_stored_sources(tmp_path):
    contract = demo_contract(tmp_path, steps=2)
    names = enabled_tools(contract)
    assert "read_source" in names and "find_source" in names
    assert "search_literature" not in names
    assert "edit_algorithm" not in names


def test_disabled_web_dispatch_makes_no_network_call(tmp_path, monkeypatch):
    contract = demo_contract(tmp_path, steps=2)
    contract.research_access.enabled = True
    agent = Harness(contract, tmp_path / "run")
    try:
        monkeypatch.setattr(agent.research, "search_web", lambda **kwargs: pytest.fail("Called disabled service"))
        with pytest.raises(PermissionError):
            agent._execute(Decision(summary="forbidden", action={"kind": "search_web", "query": "IQL"}))
    finally:
        agent.close()


def test_final_estimate_uses_all_tasks_and_new_seeds(tmp_path):
    contract = protocol_fixture(tmp_path, cases=9)
    campaign, output = tmp_path / "development", tmp_path / "final"
    freeze_fixture(contract, campaign)
    preview = audit_campaign(campaign, output, split="final", dry_run=True)
    assert preview["required"]["trials"] == 36
    assert preview["required"]["optimizer_steps"] == 36 * 2 * 3
    assert preview["protocol"]["development_tasks_in_final"] == ["fixture-0"]
    assert not output.exists()


@pytest.mark.parametrize("kwargs", [{"train_seeds": [50]}, {"eval_seeds": [50]}, {"optimizer_step_limit": 999}])
def test_final_call_cannot_change_frozen_protocol(tmp_path, kwargs):
    contract = protocol_fixture(tmp_path)
    freeze_fixture(contract, tmp_path / "dev")
    with pytest.raises(ValueError, match="overrides"):
        audit_campaign(tmp_path / "dev", tmp_path / "final", split="final", **kwargs)
    assert not (tmp_path / "final").exists()


def test_final_publishes_nine_rows_and_exposes_no_scores_to_researcher(tmp_path, stub):
    contract = protocol_fixture(tmp_path, cases=9)
    freeze_fixture(contract, tmp_path / "development")
    result = audit_campaign(tmp_path / "development", tmp_path / "final", split="final")
    assert result["comparison"]["contract_panel_complete"]
    assert result["development_tasks_in_final"] == ["fixture-0"]
    assert len(result["unseen_tasks"]) == 8
    assert len(stub.calls) == 36
    assert all(kwargs["observer"] == "independent_auditor" for _, kwargs in stub.calls)
    assert {call.train_seed for call, _ in stub.calls} == {41, 43}
    study = study_report(tmp_path, contract)
    assert len(study["rows"]) == 9 and study["research_complete"]
    assert study["scope"] == "protocol_test_NOT_D4RL"
    assert not study["method_improved"]
    assert len((tmp_path / "results.csv").read_text().splitlines()) == 10
    agent = Harness(contract, tmp_path / "development")
    try:
        agent._initialize()
        context = agent._context()
        assert "audit.json" not in json.dumps(context)
        assert len(agent.store.trials()) == 2  # Only development fixture facts.
    finally:
        agent.close()


def test_partial_final_never_reports_full_conclusion_and_resume_reuses_completed(tmp_path, stub):
    contract = protocol_fixture(tmp_path, cases=9)
    freeze_fixture(contract, tmp_path / "development")
    stub.fail_on = "fixture-8"
    with pytest.raises(RuntimeError):
        audit_campaign(tmp_path / "development", tmp_path / "final", split="final")
    partial = study_report(tmp_path, contract)
    assert not partial["research_complete"] and partial["comparison"] is None
    assert all(r["delta"] is None for r in partial["rows"])
    assert read_json(tmp_path / "final/progress.json")["status"] == "incomplete"
    assert not (tmp_path / "final/audit.json").exists()
    count = len(stub.calls)
    stub.fail_on = None
    audit_campaign(tmp_path / "development", tmp_path / "final", split="final")
    assert len(stub.calls) == 39  # 36 unique successes + three bounded failed attempts; no completed reruns.
    assert count > 1
    audit_campaign(tmp_path / "development", tmp_path / "final", split="final")
    assert len(stub.calls) == 39  # Completed audit is idempotent, including the bounded attempts.
    assert study_report(tmp_path, contract)["research_complete"]


def test_retaining_IQL_does_not_double_compute(tmp_path, stub):
    contract = protocol_fixture(tmp_path, cases=9)
    freeze_fixture(contract, tmp_path / "development", distinct=False)
    result = audit_campaign(tmp_path / "development", tmp_path / "final", split="final")
    assert len(stub.calls) == 18
    assert result["selected_candidate"] == result["baseline_candidate"]
    assert result["comparison"]["macro_delta"] == 0


def test_final_results_cannot_be_rebound_to_another_workspace(tmp_path, stub):
    contract = protocol_fixture(tmp_path)
    freeze_fixture(contract, tmp_path / "development")
    audit_campaign(tmp_path / "development", tmp_path / "final", split="final")
    count = len(stub.calls)
    with pytest.raises(ValueError):
        audit_campaign(tmp_path / "development", tmp_path / "another-final", split="final")
    assert len(stub.calls) == count


def test_insufficient_final_budget_starts_zero_trials(tmp_path, stub):
    raw = protocol_fixture(tmp_path).model_dump(mode="json")
    raw["final_evaluation"]["max_optimizer_steps"] = 1
    contract = ResearchContract.model_validate(raw)
    freeze_fixture(contract, tmp_path / "development")
    with pytest.raises(ValueError, match="authorized"):
        audit_campaign(tmp_path / "development", tmp_path / "final", split="final")
    assert not stub.calls
    assert not (tmp_path / "final").exists()


@pytest.mark.parametrize("endpoint", ["http://remote.example/v1", "https://user:key@api.example/v1", "https://api.example/v1?key=x", "file:///etc/passwd"])
def test_model_endpoint_cannot_embed_credentials_or_use_remote_http(endpoint):
    with pytest.raises(ValueError):
        validate_endpoint(endpoint)


def test_model_config_saved_without_key(tmp_path, monkeypatch):
    monkeypatch.setenv("ARLAB_API_KEY", "TEST_SECRET_MUST_NOT_BE_SAVED")
    monkeypatch.setattr("sys.stdin", io.StringIO())
    args = argparse.Namespace(action="run", provider="compatible", model="test-model", base_url="https://api.example/v1")
    config = model_settings(tmp_path, args, ModelConfig())
    assert config.model == "test-model"
    assert "TEST_SECRET" not in (tmp_path / "model.json").read_text()
    assert (tmp_path / "model.json").stat().st_mode & 0o777 == 0o600
    assert not require_key(config, False)


def test_loopback_without_key_requires_explicit_flag(monkeypatch):
    monkeypatch.delenv("ARLAB_API_KEY", raising=False)
    monkeypatch.setattr("sys.stdin", io.StringIO())
    config = ModelConfig(provider="compatible", model="local", base_url="http://127.0.0.1:8000/v1")
    with pytest.raises(ValueError):
        require_key(config, False)
    assert not require_key(config, True)


def test_agent_cannot_run_final_cases(tmp_path):
    contract = protocol_fixture(tmp_path)
    agent = Harness(contract, tmp_path / "development")
    try:
        agent._initialize()
        with pytest.raises(PermissionError):
            agent._execute(Decision(summary="not authorized", action=RunAction(kind="run",
                candidate=agent.store.get("baseline"), case="fixture-1", train_seed=0, steps=2)))
    finally:
        agent.close()


@pytest.mark.integration
def test_real_three_task_search_then_nine_task_final_on_synthetic_fixture(tmp_path):
    """Real CPU IQL and rollout, tiny synthetic tasks, not D4RL benchmark validation."""
    from arlab.classroom.session import run_study
    contract = protocol_fixture(tmp_path / "inputs", cases=9, steps=2, seeds=(41,), dev_count=3)
    report = run_study(tmp_path / "study", contract)
    assert report["research_complete"] and report["scope"] == "protocol_test_NOT_D4RL"
    final = read_json(tmp_path / "study/final/audit.json")
    assert len(final["trials"]) == 9
    assert len(final["unseen_tasks"]) == 6
    assert len(final["development_tasks_in_final"]) == 3
    assert {t["steps"] for t in final["trials"]} == {2}


def test_exploration_covers_three_environments_and_three_dataset_regimes(tmp_path):
    """Literal expected IDs prevent an all-medium regression hidden by shared constants."""
    expected = (
        "halfcheetah-medium-v2",
        "hopper-medium-replay-v2",
        "walker2d-medium-expert-v2",
    )
    assert EXPLORATION_TASKS == expected
    axes = [name.removesuffix("-v2").split("-", 1) for name in expected]
    assert {env for env, _ in axes} == {"halfcheetah", "hopper", "walker2d"}
    assert {regime for _, regime in axes} == {"medium", "medium-replay", "medium-expert"}
    contract = nine_contract(tmp_path)
    assert [c.name for c in contract.cases if c.split == "dev"] == list(expected)
    assert all(name in contract.goal for name in expected)
    assert "confounded" in contract.goal
    assert budget_preview(contract)["development_cells_per_method_per_rung"] == 9


@pytest.mark.parametrize("devs", [
    ("hopper-medium-v2",),
    ("halfcheetah-medium-v2", "hopper-medium-v2", "walker2d-medium-v2"),
    ("hopper-medium-v2", "hopper-medium-replay-v2", "hopper-medium-expert-v2"),
])
def test_student_entry_rejects_single_or_nondiverse_development_panel(tmp_path, devs):
    raw = nine_contract(tmp_path).model_dump(mode="json")
    for case in raw["cases"]:
        case["split"] = "dev" if case["name"] in devs else "test"
    with pytest.raises(ValueError, match="mixed-distribution panel"):
        require_nine_task_protocol(ResearchContract.model_validate(raw))


@pytest.mark.parametrize("field,value", [("train_seeds", [0]), ("rungs", [50_000])])
def test_full_three_task_development_cannot_be_shrunk_to_smoke(tmp_path, field, value):
    raw = nine_contract(tmp_path).model_dump(mode="json")
    raw[field] = value
    with pytest.raises(ValueError, match="Full development|Confirmation seeds"):
        require_nine_task_protocol(ResearchContract.model_validate(raw))


def three_dev_fixture(tmp_path):
    raw = protocol_fixture(tmp_path, cases=9, dev_count=3).model_dump(mode="json")
    raw["train_seeds"] = [0, 1, 2]
    return ResearchContract.model_validate(raw)


def add_observed_panel(agent, cid, *, missing=None, scores=(0.0, 0.0, 0.0), purpose="experiment"):
    """Synthetic ledger helper: no claim that these values were measured."""
    dev = [c for c in agent.contract.cases if c.split == "dev"]
    for i, case in enumerate(dev):
        for seed in agent.contract.train_seeds:
            if missing == (i, seed):
                continue
            tid = f"panel-fixture-{cid}-{case.name}-{seed}-{purpose}"
            agent.store.save_trial(tid, cid, "complete", {
                "trial_id": tid, "candidate": cid, "case": case.name, "train_seed": seed,
                "steps": agent.contract.rungs[-1], "metric": "synthetic_fixture",
                "score": scores[i], "purpose": purpose, "evidence_type": "ledger_fixture_NOT_training",
            })


@pytest.mark.parametrize("role", ["baseline", "candidate"])
@pytest.mark.parametrize("missing_case", [0, 1, 2])
def test_missing_one_cell_on_any_development_task_blocks_freeze(tmp_path, role, missing_case):
    contract = three_dev_fixture(tmp_path / "input")
    agent = Harness(contract, tmp_path / "run")
    try:
        agent._initialize()
        baseline = agent.store.get("baseline")
        recipe = copy.deepcopy(contract.seed_recipe)
        recipe["name"] = "negative-result-fixture"
        candidate = agent.store.add_candidate(recipe, {"question": "controlled fixture"}, baseline)
        add_observed_panel(agent, baseline, missing=(missing_case, 2) if role == "baseline" else None)
        add_observed_panel(agent, candidate, missing=(missing_case, 2) if role == "candidate" else None,
                           scores=(-1.0, -1.0, -1.0))
        action = Decision(summary="attempt freeze", action=FinishAction(
            kind="finish", candidate=candidate, conclusion="Fixture negative result."))
        with pytest.raises(ValueError):
            agent._execute(action)
        assert agent.store.get("frozen") is None
        assert agent.store.get("status") != "completed"
        # Completing just the missing evidence enables even a negative-result conclusion.
        add_observed_panel(agent, baseline)
        add_observed_panel(agent, candidate, scores=(-1.0, -1.0, -1.0))
        agent._execute(action)
        assert agent.store.get("status") == "completed"
    finally:
        agent.close()


def test_three_task_ranking_uses_all_tasks_and_checks_never_count(tmp_path):
    from arlab.core.panels import PanelIndex
    contract = three_dev_fixture(tmp_path / "input")
    agent = Harness(contract, tmp_path / "run")
    try:
        agent._initialize()
        baseline = agent.store.get("baseline")
        recipe = copy.deepcopy(contract.seed_recipe)
        recipe["name"] = "one-task-winner-fixture"
        candidate = agent.store.add_candidate(recipe, {"question": "fixture"}, baseline)
        add_observed_panel(agent, baseline)
        add_observed_panel(agent, candidate, scores=(2.0, -3.0, -3.0))
        add_observed_panel(agent, candidate, scores=(1000.0, 1000.0, 1000.0), purpose="check")
        index = PanelIndex(contract, agent.store.trials())
        rank = index.ranking([baseline, candidate], contract.rungs[-1])
        assert rank[0]["candidate"] == baseline
        assert rank[1]["mean"] == pytest.approx(-4 / 3)
        assert rank[1]["completed"] == 9
        comparison = index.compare(baseline, candidate, contract.rungs[-1])
        assert comparison["contract_panel_complete"]
        assert len(comparison["per_task_delta"]) == 3
        context = agent._context()
        assert context["protocol"]["required_cells_per_rung"] == 9
        assert context["confirmation"]["development_tasks_in_final_count"] == 3
        assert context["confirmation"]["not_used_for_search_task_count"] == 6
        assert context["check_count"] == 9
    finally:
        agent.close()


def test_final_report_marks_three_developed_and_six_not_used_for_search(tmp_path, stub):
    contract = three_dev_fixture(tmp_path / "inputs")
    freeze_fixture(contract, tmp_path / "development")
    audit = audit_campaign(tmp_path / "development", tmp_path / "final", split="final")
    assert audit["development_task_comparison"]["pairs"] == 6
    assert audit["unseen_task_comparison"]["pairs"] == 12
    assert len(audit["development_tasks_in_final"]) == 3
    assert len(audit["unseen_tasks"]) == 6
    report = study_report(tmp_path, contract)
    assert report["research_complete"]
    assert sum(r["used_for_development"] for r in report["rows"]) == 3
    assert len(report["not_used_for_search_tasks"]) == 6
    markdown = (tmp_path / "report.md").read_text()
    assert "3 个曾用于开发" in markdown and "6 个未用于本次搜索" in markdown
    assert "开发任务复验平均差值" in markdown
    agent = Harness(contract, tmp_path / "development")
    try:
        agent._initialize()
        assert len(agent.store.trials()) == 18  # Only developer-controlled fixtures.
        assert "audit.json" not in json.dumps(agent._context())
    finally:
        agent.close()


def test_three_task_cli_budget_preview_has_sufficient_basic_capacity(tmp_path):
    from arlab.cli import main
    path = tmp_path / "contract.json"
    assert main(["init-d4rl", "--suite9", "--data-root", str(tmp_path / "absent-data"),
                 "--output", str(path), "--device", "cpu"]) == 0
    contract = ResearchContract.model_validate(read_json(path))
    require_nine_task_protocol(contract)
    preview = budget_preview(contract)
    capacity = preview["development_all_candidates_all_rungs"]
    assert contract.max_candidates == 128 and contract.max_revisions == 10000
    assert contract.budget.max_trials == 20000
    assert preview["adaptive_start"] == {"tasks": 1, "seeds": 1, "steps": 5000, "eval_episodes": 3}
    assert capacity["trials"] == 128 * 9 * 6
    assert capacity["ticks"] == 128 * 9 * 1_000_000  # Shared trajectory, not a sum of restarts.
    assert capacity["ticks"] > contract.budget.max_train_steps  # Exhaustive search deliberately NOT promised.
    assert "NOT a schedule" in capacity["scope"]
    assert preview["development_final_rung_runs_for_two_methods"] == 18
    assert preview["final_runs_if_candidate_differs"] == 90


def test_redeploy_rejects_changed_development_roles_without_overwrite(tmp_path, monkeypatch):
    from arlab.deployment.tasks import configure
    new_contract = nine_contract(tmp_path / "identity")
    old_raw = new_contract.model_dump(mode="json")
    for case in old_raw["cases"]:
        case["split"] = "dev" if case["name"] == "hopper-medium-v2" else "test"
    old = ResearchContract.model_validate(old_raw)
    write_json(tmp_path / "configs/research.json", old)
    write_json(tmp_path / "settings.json", {})
    monkeypatch.setattr("arlab.deployment.tasks.make_contract", lambda _: new_contract)
    with pytest.raises(ValueError, match="conflicts"):
        configure(tmp_path)
    assert read_json(tmp_path / "configs/research.json") == old.model_dump(mode="json")
    assert read_json(tmp_path / "configs/research.proposed.json") == new_contract.model_dump(mode="json")
