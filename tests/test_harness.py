from __future__ import annotations

import os
from pathlib import Path

import pytest

from arlab.cli import demo_contract
from arlab.core.harness import Harness
from arlab.core.io import canonical
from arlab.core.schema import Decision, FinishAction, ResearchContract, RunAction


def test_context_contains_no_heldout_cases(tmp_path):
    contract = demo_contract(tmp_path, steps=3, candidates=2)
    agent = Harness(contract, tmp_path / "run")
    try:
        agent._initialize()
        context = canonical(agent._context())
        assert "lq-test" not in context and "lq-validation" not in context
        assert str(tmp_path / "data/test.npz") not in context
    finally:
        agent.close()


def test_researcher_cannot_run_test_or_change_training_budget(tmp_path):
    contract = demo_contract(tmp_path, steps=3, candidates=2)
    agent = Harness(contract, tmp_path / "run")
    try:
        agent._initialize()
        cid = agent.store.get("baseline")
        for action in [RunAction(kind="run", candidate=cid, case="lq-test", train_seed=0, steps=3),
                       RunAction(kind="run", candidate=cid, case="lq-dev", train_seed=42, steps=3),
                       RunAction(kind="run", candidate=cid, case="lq-dev", train_seed=0, steps=999)]:
            with pytest.raises(PermissionError):
                agent.service.run(action)
        assert not agent.store.totals()
    finally:
        agent.close()


def test_finish_requires_complete_final_panel(tmp_path):
    contract = demo_contract(tmp_path, steps=3)
    agent = Harness(contract, tmp_path / "run")
    try:
        agent._initialize()
        with pytest.raises(ValueError, match="complete"):
            agent._execute(Decision(summary="premature", action=FinishAction(kind="finish", candidate=agent.store.get("baseline"), conclusion="not measured")))
        assert agent.store.get("frozen") is None
    finally:
        agent.close()


@pytest.mark.integration
def test_full_real_campaign_and_idempotent_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1] / "src"))
    contract = demo_contract(tmp_path, steps=4, candidates=2)
    root = tmp_path / "run"
    agent = Harness(contract, root)
    try:
        result = agent.run()
        assert result["status"] == "completed", agent.store.events(20)
        assert result["trial_counts"] == {"complete": 2}
        assert result["memory_count"] >= 1
        assert all(t["metric"] == "synthetic_raw_return" for t in agent.store.trials())
        before = agent.store.totals()
        actions = agent.store.count_actions()
    finally:
        agent.close()
    def forbidden(*args, **kwargs):
        raise AssertionError("A completed campaign must not retrain")
    from arlab.core.experiments import TrialService
    monkeypatch.setattr(TrialService, "_worker", forbidden)
    agent = Harness(contract, root)
    try:
        assert agent.run()["status"] == "completed"
        assert agent.store.totals() == before
        assert agent.store.count_actions() == actions
    finally:
        agent.close()


def test_unknown_generic_task_requires_explicit_installation():
    from arlab.tasks.protocol import get_task
    with pytest.raises(ValueError):
        get_task("__import__('os')")
