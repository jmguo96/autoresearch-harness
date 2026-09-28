"""Protocol regressions. Large ledgers below are fixtures, NOT benchmark results."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import pytest
from arlab.cli import demo_contract
from arlab.core.config import AgentConfig
from arlab.core.action_history import record_research_journal, rebuild_research_journal
from arlab.core.harness import Harness
from arlab.core.io import canonical, digest
from arlab.core.panels import PanelIndex
from arlab.core.paging import text_page
from arlab.core.schema import (Decision, Feedback, FinishAction, CompareAction, QueryExperimentsAction,
                               InspectAction, CheckCandidateAction, ContinueTrialAction, ResearchContract)
from arlab.core.stats import paired_comparison


def record(agent, cid, case, seed, steps, score=1.0, purpose="experiment", diagnostic_size=0):
    tid = "fixture-" + digest([cid, case, seed, steps, purpose])[:24]
    value = {"trial_id": tid, "candidate": cid, "case": case, "train_seed": seed, "steps": steps,
             "score": score, "metric": "FIXTURE_not_D4RL", "returns": [score] * 10,
             "purpose": purpose, "diagnostics": {"large": "do_not_replay" * diagnostic_size}}
    agent.store.save_trial(tid, cid, "complete", value)
    return tid


def add_candidate(agent, i):
    recipe = copy.deepcopy(agent.contract.seed_recipe)
    recipe["name"] = f"candidate-{i}"
    return agent.store.add_candidate(recipe, {"question": "Fixture hypothesis " + str(i)}, agent.store.get("baseline"))


def test_context_excludes_repeated_schema_baseline_code_and_full_ledger(tmp_path):
    contract = demo_contract(tmp_path, steps=3, candidates=2)
    contract.model.provider = "compatible"
    agent = Harness(contract, tmp_path / "campaign")
    try:
        agent._initialize()
        record(agent, agent.store.get("baseline"), "lq-dev", 0, 3, diagnostic_size=20000)
        context = agent._context()
        text = canonical(context)
        assert "do_not_replay" not in text
        assert "trials" not in context and "recipe_schema" not in context and "baseline_recipe" not in context
        assert agent.config.instruction not in text
        assert "lq-test" not in text and "lq-validation" not in text
        assert context["panels"][0]["complete"]
        assert len(text) <= agent.config.context_characters
    finally:
        agent.close()


@pytest.mark.parametrize("candidate_count", [4, 20])
def test_large_200_and_1000_record_context_is_bounded_and_queryable(tmp_path, candidate_count):
    contract = demo_contract(tmp_path, steps=2, candidates=candidate_count)
    raw = contract.model_dump(mode="json")
    raw["model"]["provider"] = "compatible"
    raw["cases"] = [{**raw["cases"][0], "name": f"task-{i}"} for i in range(6)]
    raw["train_seeds"] = [0, 1, 2]
    raw["rungs"] = [2, 4, 8]
    agent = Harness(ResearchContract.model_validate(raw), tmp_path / "run")
    try:
        agent._initialize()
        ids = [agent.store.get("baseline")] + [add_candidate(agent, i) for i in range(candidate_count - 1)]
        for cid in ids:
            for case in raw["cases"]:
                for seed in raw["train_seeds"]:
                    for step in raw["rungs"]:
                        record(agent, cid, case["name"], seed, step, diagnostic_size=100)
        count = len(agent.store.trials())
        assert count in (216, 1080)
        context = agent._context()
        assert len(canonical(context)) < 28000
        assert context["trial_counts"]["complete"] == count
        assert context["ranking_steps"] == 8
        page = agent._execute(Decision(summary="query facts", action=QueryExperimentsAction(
            kind="query_experiments", limit=30)))
        assert len(page.details["items"]) == 30 and page.details["total"] == count
        assert page.details["next_offset"] == 30
        assert agent.store.get("status") != "budget_exhausted"
    finally:
        agent.close()


def test_huge_feedback_remains_retrievable_without_context_failure(tmp_path):
    agent = Harness(demo_contract(tmp_path, steps=2), tmp_path / "run")
    try:
        agent._initialize()
        feedback = {"status": "ok", "summary": "Full source received", "details": {"text": "x" * 100000}, "evidence_ids": []}
        ref = agent.store.save_artifact("feedback", json.dumps(feedback, indent=2), {})
        agent.store.put("last_feedback", feedback)
        agent.store.put("last_feedback_ref", ref)
        context = agent._context()
        assert context["last_feedback"]["truncated"] and context["last_feedback"]["artifact_ref"] == ref
        assert len(canonical(context)) < 28000
        detail = agent._execute(Decision(summary="read detail", action=InspectAction(kind="inspect", target="feedback", id=ref)))
        assert detail.details["truncated"]
        assert agent.store.artifact(ref)["text"] == json.dumps(feedback, indent=2)
    finally:
        agent.close()


def _persist_action(agent, decision, feedback):
    action_id = agent.store.begin_action(decision)
    agent.store.finish_action(action_id, feedback)
    agent.store.put("last_feedback", feedback.model_dump(mode="json"))
    return action_id


def test_context_retains_distinct_tool_results_and_rejects_duplicate_reads(tmp_path):
    config = AgentConfig(max_observations_without_progress=4)
    agent = Harness(demo_contract(tmp_path, steps=2), tmp_path / "run", config=config)
    try:
        agent._initialize()
        baseline = agent.store.get("baseline")
        decisions = [
            Decision(summary="candidate", action=InspectAction(
                kind="inspect", target="candidate", id=baseline, max_lines=60)),
            Decision(summary="recipe schema", action=InspectAction(
                kind="inspect", target="schema", id="recipe")),
            Decision(summary="decision schema", action=InspectAction(
                kind="inspect", target="schema", id="decision")),
            Decision(summary="run tool", action=InspectAction(
                kind="inspect", target="tool", id="run")),
        ]
        for decision in decisions:
            _persist_action(agent, decision, agent._execute(decision))
        context = agent._context()
        retained = {x["action"]["id"] for x in context["recent_tool_results"]}
        assert baseline in retained and "recipe" in retained
        assert context["action_counts"]["inspect"] == 4
        assert context["loop_guard"]["remaining_before_progress_required"] == 0

        duplicate = Decision(summary="reread", action=InspectAction(
            kind="inspect", target="candidate", id=baseline, max_lines=150))
        feedback = agent._observation_guard(duplicate)
        assert feedback and feedback.status == "invalid_action"
        assert feedback.details["reason"] == "duplicate_observation"
        assert feedback.details["read_prior_result"].endswith("id='action-1')")

        novel = Decision(summary="more reading", action=InspectAction(
            kind="inspect", target="skill", id="iql"))
        feedback = agent._observation_guard(novel)
        assert feedback and feedback.details["reason"] == "observation_phase_limit"
    finally:
        agent.close()


def test_research_journal_persists_and_is_rebuildable_from_fact_ledger(tmp_path):
    agent = Harness(demo_contract(tmp_path, steps=2), tmp_path / "run")
    try:
        agent._initialize()
        baseline = agent.store.get("baseline")
        decision = Decision(summary="Inspect baseline once", action=InspectAction(
            kind="inspect", target="candidate", id=baseline))
        feedback = agent._execute(decision)
        action_id = agent.store.begin_action(decision)
        agent.store.finish_action(action_id, feedback)
        record_research_journal(agent.store, action_id, decision, feedback)

        journal = agent.store.get("research_journal")
        assert journal["last_action_id"] == action_id
        assert journal["action_counts"]["inspect"] == 1
        assert len(journal["observations"]) == 1
        context = agent._context()
        assert context["research_progress"]["last_recorded_action"] == action_id
        assert context["research_progress"]["observation_inventory"]["unique"] == 1

        rebuilt = rebuild_research_journal(agent.store)
        assert rebuilt["action_counts"] == journal["action_counts"]
        assert rebuilt["observations"].keys() == journal["observations"].keys()
        assert next(iter(rebuilt["observations"].values()))["feedback_ref"] == f"action-{action_id}"
        retrieved = agent._execute(Decision(summary="recover old result", action=InspectAction(
            kind="inspect", target="feedback", id=f"action-{action_id}")))
        assert json.loads(retrieved.details["text"])["summary"] == feedback.summary
    finally:
        agent.close()


def test_old_action_feedback_is_retrievable_after_recent_window(tmp_path):
    agent = Harness(demo_contract(tmp_path, steps=2), tmp_path / "run")
    try:
        agent._initialize()
        decision = Decision(summary="first result", action=InspectAction(
            kind="inspect", target="skill", id="iql"))
        feedback = Feedback(status="ok", summary="retained historical result",
                            details={"text": "early evidence"})
        first_id = _persist_action(agent, decision, feedback)
        for index in range(110):
            _persist_action(agent, Decision(summary=f"plan {index}", action=InspectAction(
                kind="inspect", target="plan")), Feedback(status="ok", summary="later"))
        assert not any(x["action_id"] == first_id for x in agent._context()["recent_tool_results"])
        response = agent._execute(Decision(summary="recover", action=InspectAction(
            kind="inspect", target="feedback", id=f"action-{first_id}")))
        assert "early evidence" in response.details["text"]
        assert agent._observation_guard(decision).details["first_successful_action"] == first_id
    finally:
        agent.close()


def test_control_source_migration_keeps_old_trial_as_evidence_not_checkpoint(tmp_path):
    agent = Harness(demo_contract(tmp_path, steps=2), tmp_path / "run")
    try:
        agent._initialize()
        baseline = agent.store.get("baseline")
        old_trial = record(agent, baseline, "lq-dev", 0, 2)
        agent.store.put("source_migration", {
            "legacy_trials": [old_trial],
            "note": "Old trial remains evidence; checkpoint cannot continue under new source.",
        })
        context = agent._context()
        assert old_trial not in {x["trial_id"] for x in context["recent_partial_observations"]}
        assert old_trial not in {x["trial_id"] for x in context["continuable_trials"]}
        assert context["source_migration"]["legacy_trials"] == [old_trial]
        assert agent.store.trial(old_trial)["status"] == "complete"
        assert not agent.store.panel_trials()
        with pytest.raises(ValueError, match="old source checkpoint"):
            agent.service.continue_trial(ContinueTrialAction(kind="continue_trial", trial=old_trial, steps=3))
        current_trial = {**agent.store.trial(old_trial), "trial_id": "current-fixture",
                         "source_sha256": agent.store.get("source_sha256")}
        agent.store.save_trial("current-fixture", baseline, "complete", current_trial)
        assert agent._context()["panels"][0]["completed"] == 1
    finally:
        agent.close()


@pytest.mark.parametrize("text", ["", "one\ntwo\nthree", "x" * 18000, "中文" * 8000 + "\nend\n", "\n\n"])
def test_lossless_paging_including_single_long_line(text):
    cursor, parts = {"start_line": 1, "offset": 0}, []
    for _ in range(100):
        page = text_page(text, max_lines=2, max_characters=777, **cursor)
        assert len(page["text"]) <= 777
        parts.append(page["text"])
        cursor = page["next"]
        if cursor is None:
            break
    assert "".join(parts) == text


def test_finish_requires_baseline_even_when_candidate_complete_and_accepts_negative_result(tmp_path):
    agent = Harness(demo_contract(tmp_path, steps=2), tmp_path / "run")
    try:
        agent._initialize()
        cid = add_candidate(agent, 1)
        record(agent, cid, "lq-dev", 0, 2, score=-1)
        finish = Decision(summary="freeze", action=FinishAction(kind="finish", candidate=cid, conclusion="Negative result"))
        with pytest.raises(ValueError, match="baseline"):
            agent._execute(finish)
        record(agent, agent.store.get("baseline"), "lq-dev", 0, 2, score=1)
        agent._execute(finish)
        assert agent.store.get("status") == "completed"
    finally:
        agent.close()


def test_check_never_qualifies_as_complete_final_panel(tmp_path):
    agent = Harness(demo_contract(tmp_path, steps=2), tmp_path / "run")
    try:
        agent._initialize()
        cid = agent.store.get("baseline")
        record(agent, cid, "lq-dev", 0, 2, purpose="check")
        assert agent._ranking(final=True) == []
        assert agent._context()["check_count"] == 1
    finally:
        agent.close()


def test_contract_coverage_not_matching_incomplete_subsets():
    rows = [{"case": c, "train_seed": s, "steps": 10, "metric": "fixture", "score": 1}
            for c in ["a", "b"] for s in [0, 1]]
    expected = {(c, s, 10) for c in ["a", "b", "c", "d", "e", "f"] for s in [0, 1, 2]}
    result = paired_comparison(rows, [{**r, "score": 2} for r in rows], expected_cells=expected)
    assert not result["complete"] and result["matched_observed_cells"]
    assert result["coverage"]["expected"] == 18
    assert len(result["coverage"]["missing_baseline"]) == 14
    assert result["status"] == "insufficient_evidence"


def test_latest_comparable_rung_changes_rank_without_mixing_fidelities(tmp_path):
    contract = demo_contract(tmp_path, steps=2)
    contract.rungs = [2, 4, 8]
    agent = Harness(contract, tmp_path / "run")
    try:
        agent._initialize()
        base = agent.store.get("baseline")
        cid = add_candidate(agent, 1)
        for candidate, step, score in [(base, 2, 0), (cid, 2, 10), (base, 4, 10), (cid, 4, -100), (base, 8, 20)]:
            record(agent, candidate, "lq-dev", 0, step, score)
        ranks = agent._ranking(final=False)
        assert ranks[0]["candidate"] == base
        assert {r["steps"] for r in ranks} == {4}
    finally:
        agent.close()


def test_missing_query_does_not_expose_test_cases(tmp_path):
    agent = Harness(demo_contract(tmp_path, steps=2), tmp_path / "run")
    try:
        agent._initialize()
        action = QueryExperimentsAction(kind="query_experiments", target="missing")
        result = agent._execute(Decision(summary="coverage", action=action))
        assert {x["case"] for x in result.details["items"]} == {"lq-dev"}
        with pytest.raises(PermissionError):
            agent._execute(Decision(summary="unauthorized", action=QueryExperimentsAction(
                kind="query_experiments", case="lq-test")))
    finally:
        agent.close()


def test_rsi_commands_and_imports_removed():
    from arlab.cli import build_parser
    import importlib.util
    assert importlib.util.find_spec("arlab.evolution") is None
    help_text = build_parser().format_help()
    assert "rsi-demo" not in help_text and "compare-improvers" not in help_text
    with pytest.raises(ValueError):
        AgentConfig.model_validate({"improver": {}})


@pytest.mark.integration
def test_real_candidate_check_charged_but_not_ranked(tmp_path):
    agent = Harness(demo_contract(tmp_path, steps=2), tmp_path / "run")
    try:
        agent._initialize()
        base = agent.store.get("baseline")
        result = agent._execute(Decision(summary="cheap check", action=CheckCandidateAction(
            kind="check_candidate", candidate=base, case="lq-dev", train_seed=0, steps=2)))
        assert result.details["purpose"] == "check"
        assert len(result.details["returns"]) == 1
        assert agent.store.totals()["checks"] == 1
        assert agent.store.totals()["train_steps"] == 2
        assert agent._ranking(final=True) == []
    finally:
        agent.close()


def test_audit_budget_preflight_accounts_for_open_candidate_before_jobs(tmp_path):
    from arlab.algorithms.templates import template
    from arlab.audit import audit_campaign
    contract = demo_contract(tmp_path, steps=10, open_code=True)
    agent = Harness(contract, tmp_path / "campaign")
    try:
        agent._initialize()
        base = agent.store.get("baseline")
        recipe = template("ensemble-mixture", tiny=True)
        recipe["max_optimizer_steps_per_tick"] = 64
        cid = agent.store.add_candidate(recipe, {"question": "cost fixture"}, base)
        for candidate in [base, cid]:
            record(agent, candidate, "lq-dev", 0, 10)
        agent._execute(Decision(summary="freeze for audit", action=FinishAction(kind="finish", candidate=cid, conclusion="fixture")))
    finally:
        agent.close()
    out = tmp_path / "audit"
    preview = audit_campaign(tmp_path / "campaign", out, train_seeds=[41, 43], eval_seeds=[9], dry_run=True)
    assert preview["required"]["optimizer_steps"] == 1 * 2 * 10 * (3 + 64)
    assert not out.exists()
    with pytest.raises(ValueError, match="before starting"):
        audit_campaign(tmp_path / "campaign", out, train_seeds=[41, 43], optimizer_step_limit=1)
    assert not out.exists()


def test_source_linked_memory_not_starved_by_trial_observations(tmp_path):
    agent = Harness(demo_contract(tmp_path, steps=2), tmp_path / "run")
    try:
        agent._initialize()
        eid = agent.store.event("fixture", {})
        agent.store.remember("IQL value-learning source needs experimental validation", "literature", [eid])
        for i in range(100):
            agent.store.remember(f"IQL trial episode {i}", "episode", [eid])
        notes = agent._context()["memory"]
        assert any(m["category"] == "literature" for m in notes)
        assert not any(m["category"] == "episode" for m in notes)
    finally:
        agent.close()


def test_compact_schema_preserves_local_types_without_redundant_titles():
    from arlab.core.models import schema_prompt
    prompt = schema_prompt(Decision)
    assert '"title":' not in prompt and '"required":' in prompt and '"discriminator":' in prompt
    assert len(prompt) < len(canonical(Decision.model_json_schema()))


def test_owner_agent_config_cannot_be_modified_by_action():
    with pytest.raises(ValueError):
        Decision.model_validate({"summary": "change permissions", "action": {"kind": "edit_agent", "instruction": "x"}})


def test_configure_online_research_is_separate_from_candidate_code(tmp_path):
    from arlab.deployment.tasks import make_contract
    settings = {"root": str(tmp_path), "profile": "smoke", "smoke_steps": 2,
                "device": "cpu", "provider": "compatible", "model": "configured", "base_url": "http://localhost:8000/v1",
                "code_mode": "disabled", "image": None, "research_online": True,
                "research_hosts": ["docs.example.org"]}
    contract = make_contract(settings)
    assert contract.research_access.enabled
    assert "docs.example.org" in contract.research_access.allowed_hosts
    assert contract.code_execution.mode == "disabled"


def test_context_character_counter_matches_actual_serialization(tmp_path):
    agent = Harness(demo_contract(tmp_path, steps=2), tmp_path / 'run')
    try:
        agent._initialize()
        context = agent._context()
        assert context['context_control']['characters'] == len(canonical(context))
    finally:
        agent.close()


def test_unsupported_panel_filter_fails_instead_of_returning_false_empty_result(tmp_path):
    agent = Harness(demo_contract(tmp_path, steps=2), tmp_path / 'run')
    try:
        agent._initialize()
        with pytest.raises(ValueError, match='case filtering'):
            agent._execute(Decision(summary='read panel', action=QueryExperimentsAction(
                kind='query_experiments', target='panels', case='lq-dev')))
        agent.store.save_trial('failure', agent.store.get('baseline'), 'failed', {
            'trial_id': 'failure', 'candidate': agent.store.get('baseline'), 'case': 'lq-dev',
            'train_seed': 0, 'steps': 2, 'purpose': 'experiment'})
        result = agent._execute(Decision(summary='read missing failures', action=QueryExperimentsAction(
            kind='query_experiments', target='missing', status='failed')))
        assert len(result.details['items']) == 1
        assert result.details['items'][0]['status'] == 'failed'
    finally:
        agent.close()


@pytest.mark.parametrize('error', ['xml', 'missing_key'])
def test_malformed_research_data_enters_tool_failure_boundary(tmp_path, error):
    from xml.etree.ElementTree import ParseError
    from arlab.core.schema import SearchLiteratureAction
    from arlab.research.network import ResearchError
    class MalformedProvider:
        def search_literature(self, **kwargs):
            raise ParseError('broken XML') if error == 'xml' else KeyError('message')
    contract = demo_contract(tmp_path, steps=2)
    contract.research_access.enabled = True  # Exercise malformed data AFTER owner authorization.
    agent = Harness(contract, tmp_path / 'run', research=MalformedProvider())
    try:
        agent._initialize()
        with pytest.raises(ResearchError, match='could not be decoded'):
            agent._execute(Decision(summary='search', action=SearchLiteratureAction(
                kind='search_literature', query='IQL')))
    finally:
        agent.close()
