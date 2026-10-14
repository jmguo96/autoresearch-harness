from __future__ import annotations

import pytest

from arlab.cli import demo_contract
from arlab.core.harness import Harness
from arlab.core.proposals import method_changed
from arlab.core.schema import Decision, Hypothesis, ProposeAction, ResearchContract, ReviewProposalAction


def test_scalar_iql_tuning_is_not_a_method_change(tmp_path):
    parent = demo_contract(tmp_path).seed_recipe
    tuned = {**parent, "weight_cap": 20.0, "name": "better-iql"}
    assert not method_changed(parent, tuned)
    changed = {**parent, "actor_weight": {"op": "const", "value": 1}}
    assert method_changed(parent, changed)


def test_real_model_requires_matching_approved_review_before_candidate(tmp_path):
    raw = demo_contract(tmp_path).model_dump(mode="json")
    raw["model"].update(provider="responses", model="gpt-5.6-sol",
                        base_url="https://example.com/v1")
    contract = ResearchContract.model_validate(raw)
    agent = Harness(contract, tmp_path / "run")
    try:
        agent._initialize()
        baseline = agent.store.get("baseline")
        hypothesis = Hypothesis(question="New actor objective?", prediction="May help",
                                rationale="Reduce extrapolation", falsification="No matched gain")
        recipe = {**raw["seed_recipe"], "actor_weight": {"op": "exp", "args": [{"var": "adv"}]}}
        def propose(review_id=""):
            return Decision(summary="candidate", action=ProposeAction(
                kind="propose", parent=baseline, hypothesis=hypothesis,
                recipe=recipe, review_id=review_id))
        with pytest.raises(ValueError, match="approved review_proposal"):
            agent._execute(propose())
        rejected = agent._execute(Decision(summary="review", action=ReviewProposalAction(
            kind="review_proposal", parent=baseline, hypothesis=hypothesis,
            method_change="New actor objective", why_not_tuning="Objective changes",
            feasibility="Simple expression", failure_mode="Underfit", verdict="reject")))
        with pytest.raises(ValueError, match="approved review_proposal"):
            agent._execute(propose(rejected.details["review_id"]))
        approved = agent._execute(Decision(summary="review", action=ReviewProposalAction(
            kind="review_proposal", parent=baseline, hypothesis=hypothesis,
            method_change="New actor objective", why_not_tuning="Objective changes",
            feasibility="Simple expression", failure_mode="Underfit", verdict="proceed")))
        assert agent._context()["proposal_review"]["id"] == approved.details["review_id"]
        assert agent.store.get("proposal_review")["hypothesis"]["question"] == hypothesis.question
        assert "hypothesis_sha256" not in agent.store.get("proposal_review")
        scalar = {**raw["seed_recipe"], "weight_cap": 20.0}
        with pytest.raises(ValueError, match="Only scalar hyperparameters"):
            agent._execute(Decision(summary="tuning", action=ProposeAction(
                kind="propose", parent=baseline, hypothesis=hypothesis,
                recipe=scalar, review_id=approved.details["review_id"])))
        # An equivalent public description need not be copied byte-for-byte.
        paraphrased = hypothesis.model_copy(update={"question": "Can the changed actor objective improve results?"})
        result = agent._execute(Decision(summary="candidate", action=ProposeAction(
            kind="propose", parent=baseline, hypothesis=paraphrased,
            recipe=recipe, review_id=approved.details["review_id"])))
        assert result.details["candidate"]
        assert agent.store.get("proposal_review") is None
        with pytest.raises(ValueError, match="approved review_proposal"):
            agent._execute(propose(approved.details["review_id"]))
    finally:
        agent.close()


def test_tool_rejections_do_not_count_as_model_response_failures(tmp_path, monkeypatch):
    from arlab.core.schema import Feedback
    contract = demo_contract(tmp_path, steps=3)
    agent = Harness(contract, tmp_path / "run")
    decision = Decision(summary="fixture", action={
        "kind": "inspect", "target": "skill", "id": "iql"})
    replies = iter([decision, decision, ValueError("Malformed model JSON"), decision])
    calls = 0
    def decide():
        item = next(replies)
        if isinstance(item, Exception):
            raise item
        return item
    def execute(_):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ValueError("Rejected proposal fixture")
        agent.store.put("status", "completed")
        return Feedback(status="ok", summary="Recovery fixture")
    monkeypatch.setattr(agent, "_decide", decide)
    monkeypatch.setattr(agent, "_execute", execute)
    try:
        assert agent.run()["status"] == "completed"
        assert calls == 3
        assert sum(e["kind"] == "model_failure" for e in agent.store.events()) == 1
    finally:
        agent.close()
