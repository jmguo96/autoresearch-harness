"""Regression tests for the OPEN boundary, not for every candidate looking like IQL."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from arlab.algorithms.context import RecordedData, UpdateContext
from arlab.algorithms.guard import BoundaryViolation, ValueBoundary
from arlab.algorithms.loading import load_algorithm
from arlab.algorithms.protocol import BuildSpec
from arlab.algorithms.runtime import train_code
from arlab.algorithms.spec import CodeRecipe
from arlab.algorithms.templates import template
from arlab.core.io import digest, read_json
from arlab.tasks.iql.data import OfflineReplay, load_dataset, save_dataset
from arlab.tasks.iql.evaluator import evaluate_policy
from arlab.tasks.iql.pack import IQLPack
from arlab.tasks.iql.trainer import IQLTrainer
from arlab.tasks.iql.recipe import IQLRecipe


def make_learner(dataset, tmp_path, name="ensemble-mixture", **config):
    recipe = CodeRecipe.model_validate(template(name, tiny=True))
    recipe.config.update(config)
    recipe.materialize(tmp_path / "source")
    learner = load_algorithm(recipe, tmp_path / "source", BuildSpec(3, 2, 20, "cpu", recipe.config))
    arrays, metadata = load_dataset(dataset[0])
    replay = OfflineReplay(arrays, metadata, 5)
    return recipe, learner, replay


def request(dataset, name="ensemble-mixture", steps=6):
    return {"recipe": template(name, tiny=True), "data": str(dataset[0]), "dataset_sha256": dataset[1],
            "seed": 7, "steps": steps, "device": "cpu", "code_execution": {"mode": "trusted_local"}}


def evaluation_request(root, trained, request):
    return {"artifact": str(root / "artifact"), "artifact_sha256": trained["artifact_sha256"],
            "training_binding": digest(request), "steps": request["steps"], "eval_seeds": [11, 17],
            "environment": {"kind": "synthetic_lq", "dynamics_seed": 0, "horizon": 8},
            "code_execution": {"mode": "trusted_local"}}


def test_source_validation_does_not_execute_candidate(tmp_path):
    sentinel = tmp_path / "must-not-exist"
    recipe = template("ensemble-mixture", tiny=True)
    recipe["files"]["candidate.py"] = f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('oops')\ndef build(spec): pass\n"
    IQLPack().validate_recipe(recipe)
    assert not sentinel.exists()


@pytest.mark.parametrize("path", ["../escape.py", "/tmp/escape.py", "bad\\file.py", "x.txt", "a/../../x.py"])
def test_source_paths_cannot_escape(path):
    with pytest.raises(ValueError):
        CodeRecipe(name="bad", files={path: "pass"}, entrypoint=path+":build")


def test_python_disabled_without_owner_permission(dataset, tmp_path):
    raw = request(dataset)
    raw.pop("code_execution")
    with pytest.raises(PermissionError, match="owner"):
        train_code(raw, tmp_path / "out")


def test_container_required_never_falls_back_to_host(dataset, tmp_path, monkeypatch):
    monkeypatch.delenv("ARLAB_CODE_CONTAINER", raising=False)
    raw = request(dataset)
    raw["code_execution"] = {"mode": "docker", "docker_image": "not-installed:test"}
    with pytest.raises(PermissionError, match="local host"):
        train_code(raw, tmp_path / "out")


@pytest.mark.parametrize("mutation", ["shuffle", "noise", "next-state-same-action"])
def test_value_phase_rejects_unpaired_and_generated_actions(dataset, tmp_path, mutation):
    _, learner, replay = make_learner(dataset, tmp_path)
    guard = ValueBoundary(learner)
    try:
        with guard.enter("value"):
            b = RecordedData(replay, guard, "cpu").sample(16)
            obs, actions = b.observations, b.actions
            if mutation == "shuffle":
                actions = actions.roll(1, 0)
            elif mutation == "noise":
                actions = actions + 0.001
            else:
                obs = b.next_observations
            with pytest.raises(BoundaryViolation, match="unpaired"):
                with torch.no_grad():
                    learner.value_modules["target_q"](obs, actions).detach()
    finally:
        guard.close()


def test_mutating_batch_does_not_mutate_provenance(dataset, tmp_path):
    _, learner, replay = make_learner(dataset, tmp_path)
    guard = ValueBoundary(learner)
    try:
        with guard.enter("value"):
            b = RecordedData(replay, guard, "cpu").sample(16)
            b.actions.add_(0.1)
            with pytest.raises(BoundaryViolation):
                learner.value_modules["q"](b.observations, b.actions)
    finally:
        guard.close()


def test_keyword_and_recorded_successor_queries_are_allowed(dataset, tmp_path):
    _, learner, replay = make_learner(dataset, tmp_path)
    guard = ValueBoundary(learner)
    try:
        with guard.enter("value"):
            data = RecordedData(replay, guard, "cpu")
            current = data.take([0, 7])
            positions, nxt = data.successors(current)
            assert positions.tolist() == [0]  # row 7 is a time-limit boundary
            learner.value_modules["q"](observations=nxt.observations, actions=nxt.actions)
        assert guard.counts["recorded_value_queries"] == 1
    finally:
        guard.close()


def test_policy_can_sample_and_backprop_through_frozen_q(dataset, tmp_path):
    _, learner, replay = make_learner(dataset, tmp_path, "diffusion-extraction", actor_q_guidance=0.1)
    guard = ValueBoundary(learner)
    guard.tick_remaining = 5
    values = {name: copy.deepcopy(module.state_dict()) for name, module in learner.value_modules.items()}
    before_actor = copy.deepcopy(learner.policy_modules["actor"].state_dict())
    try:
        with guard.enter("policy"):
            ctx = UpdateContext(learner, guard, replay, tick=0, device="cpu", remaining_optimizations=5)
            learner.policy_step(ctx, "extract")
        assert guard.counts["policy_generated_q_queries"] > 0
        for name, module in learner.value_modules.items():
            for key, value in module.state_dict().items():
                torch.testing.assert_close(value, values[name][key], rtol=0, atol=0)
            assert all(p.grad is None for p in module.parameters())
        assert any(not torch.equal(x, before_actor[k]) for k, x in learner.policy_modules["actor"].state_dict().items())
    finally:
        guard.close()


def test_detached_policy_q_cannot_reenter_value_objective(dataset, tmp_path):
    _, learner, replay = make_learner(dataset, tmp_path)
    guard = ValueBoundary(learner)
    guard.tick_remaining = 3
    try:
        with guard.enter("policy"):
            b = RecordedData(replay, guard, "cpu").sample(16)
            cached = learner.value_modules["q"](b.observations, b.actions + 0.02).detach().clone()
        with guard.enter("value"):
            ctx = UpdateContext(learner, guard, replay, tick=1, device="cpu", remaining_optimizations=3)
            loss = learner.value_modules["v"](ctx.batch.observations).mean() + cached.mean()
            with pytest.raises(BoundaryViolation, match="leaked"):
                ctx.optimize(loss, "v")
    finally:
        guard.close()


def test_policy_optimizer_cannot_modify_value_parameters(dataset, tmp_path):
    _, learner, replay = make_learner(dataset, tmp_path)
    guard = ValueBoundary(learner)
    guard.tick_remaining = 3
    try:
        with guard.enter("policy"):
            ctx = UpdateContext(learner, guard, replay, tick=0, device="cpu", remaining_optimizations=3)
            loss = learner.policy_modules["actor"].loss_per_sample(ctx.batch.observations, ctx.batch.actions).mean()
            with pytest.raises(BoundaryViolation, match="another phase"):
                ctx.optimize(loss, "q")
    finally:
        guard.close()


@pytest.mark.parametrize("buffer", [False, True])
def test_direct_value_mutation_is_detected_and_restored(dataset, tmp_path, buffer):
    _, learner, _ = make_learner(dataset, tmp_path)
    if buffer:
        learner.value_modules["v"].register_buffer("counter", torch.zeros(1))
        changed = learner.value_modules["v"].counter
    else:
        changed = next(learner.value_modules["v"].parameters())
    snapshot = changed.detach().clone()
    guard = ValueBoundary(learner)
    try:
        with pytest.raises(BoundaryViolation, match="protected"):
            with guard.enter("policy"), torch.no_grad():
                changed.add_(1)
        torch.testing.assert_close(changed, snapshot, rtol=0, atol=0)
    finally:
        guard.close()


def test_shared_encoder_is_value_owned(dataset, tmp_path):
    _, learner, replay = make_learner(dataset, tmp_path)
    # Structural sharing is permitted, not prohibited. Actor cannot update it.
    learner.policy_modules["actor"].shared_value = learner.value_modules["v"]
    learner.optimizers["shared"] = torch.optim.SGD(learner.value_modules["v"].parameters(), lr=0.01)
    guard = ValueBoundary(learner)
    guard.tick_remaining = 4
    try:
        with guard.enter("value"):
            ctx = UpdateContext(learner, guard, replay, tick=0, device="cpu", remaining_optimizations=4)
            ctx.optimize(learner.value_modules["v"](ctx.batch.observations).square().mean(), "shared")
        with guard.enter("policy"):
            assert all(not p.requires_grad for p in learner.policy_modules["actor"].shared_value.parameters())
    finally:
        guard.close()


def test_negative_value_objectives_are_legal_in_old_and_new_api(dataset, tmp_path):
    arrays, meta = load_dataset(dataset[0])
    old = IQLTrainer(3, 2, IQLRecipe(hidden_dims=[16], value_loss={"op": "sub", "args": [
        {"op": "square", "args": [{"var": "delta"}]}, {"const": 100}]}), 2)
    assert old.update(OfflineReplay(arrays, meta, 0).sample(16, "cpu"))["value_loss"] < 0
    raw = request(dataset, steps=2)
    raw["recipe"]["config"]["loss_offset"] = 100
    result = train_code(raw, tmp_path / "new")
    assert result["diagnostics"]["value.fit.value_loss"] < 0
    assert result["diagnostics"]["value.fit.q_loss"] < 0


def test_variable_schedule_counts_actual_optimizer_steps(dataset, tmp_path):
    raw = request(dataset, steps=6)
    result = train_code(raw, tmp_path / "new")
    assert result["optimizer_steps"] == 6 * 2 + 3


def test_declared_optimizer_budget_is_enforced(dataset, tmp_path):
    raw = request(dataset, steps=2)
    raw["recipe"]["max_optimizer_steps_per_tick"] = 1
    with pytest.raises(BoundaryViolation, match="reservation"):
        train_code(raw, tmp_path / "new")


def test_multistep_targets_do_not_cross_episode_boundaries(dataset, tmp_path):
    _, learner, replay = make_learner(dataset, tmp_path)
    guard = ValueBoundary(learner)
    try:
        with guard.enter("value"):
            data = RecordedData(replay, guard, "cpu")
            b = data.take([6, 7])
            target = data.n_step(b, 5, 0.9)
            assert target.lengths.tolist() == [2, 1]
            r = replay._data["rewards"]
            torch.testing.assert_close(target.rewards, torch.stack((r[6] + 0.9*r[7], r[7])))
            torch.testing.assert_close(target.bootstrap_discount, torch.tensor([0.9**2, 0.9]))
    finally:
        guard.close()


def test_legacy_files_require_explicit_links_for_multistep(dataset, tmp_path):
    arrays, meta = load_dataset(dataset[0])
    arrays.pop("next_indices")
    old = tmp_path / "old.npz"
    save_dataset(old, arrays, meta)
    _, learner, replay = make_learner((old, ""), tmp_path)
    guard = ValueBoundary(learner)
    try:
        with guard.enter("value"):
            data = RecordedData(replay, guard, "cpu")
            with pytest.raises(ValueError, match="reimport"):
                data.n_step(data.sample(4), 2, 0.99)
    finally:
        guard.close()


@pytest.mark.parametrize("name", ["ensemble-mixture", "diffusion-extraction"])
def test_open_export_roundtrip_is_independently_evaluated(dataset, tmp_path, name):
    raw = request(dataset, name, steps=3)
    root = tmp_path / "train"
    trained = train_code(raw, root)
    evaluated = evaluate_policy(evaluation_request(root, trained, raw), tmp_path / "evaluation")
    assert len(evaluated["returns"]) == 2
    assert evaluated["inference"]["action_calls"] == 16
    assert evaluated["inference"]["candidate_instrumentation"]["inference_generated_q_queries"] == 16
    assert evaluated["inference"]["scoring_in_candidate_process"] is False
    assert train_code(raw, root) == trained


def test_open_checkpoint_restores_networks_optimizers_rng_and_schedule(dataset, tmp_path, monkeypatch):
    raw = request(dataset, "diffusion-extraction", steps=6)
    train_code(raw, tmp_path / "full")
    import arlab.algorithms.runtime as runtime
    real_save = runtime.save_checkpoint
    def crash(path, **kwargs):
        real_save(path, **kwargs)
        if kwargs["tick"] == 3:
            raise RuntimeError("injected open-code crash")
    monkeypatch.setattr(runtime, "save_checkpoint", crash)
    with pytest.raises(RuntimeError, match="injected"):
        train_code(raw, tmp_path / "resumed")
    monkeypatch.setattr(runtime, "save_checkpoint", real_save)
    result = train_code(raw, tmp_path / "resumed")
    assert result["resumed_at"] == 3
    left = load_file(str(tmp_path / "full/artifact/weights.safetensors"))
    right = load_file(str(tmp_path / "resumed/artifact/weights.safetensors"))
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)


CUSTOM_SOURCE = '''
import torch
from torch import nn
from arlab.algorithms.protocol import Algorithm, Update

class QuantileRoot(nn.Module):
    def __init__(self, obs, act):
        super().__init__()
        self.linear = nn.Linear(obs + act, 5)
    def forward(self, observations, actions):
        return self.linear(torch.cat((observations, actions), -1))

class DifferentArchitecture(Algorithm):
    def __init__(self, spec):
        super().__init__()
        self.batch_size = 8
        self.value_modules = {"quantile_q": QuantileRoot(spec.observation_dim, spec.action_dim)}
        self.action_value_names = ["quantile_q"]
        self.policy_modules = {"deterministic": nn.Sequential(nn.Linear(spec.observation_dim, spec.action_dim), nn.Tanh())}
        self.optimizers = {"value": torch.optim.SGD(self.value_modules["quantile_q"].parameters(), lr=0.01),
                           "policy": torch.optim.SGD(self.policy_modules["deterministic"].parameters(), lr=0.01)}
    def schedule(self, tick):
        return [Update("policy", "bc"), Update("value", "vector-regression")]
    def value_step(self, ctx, name):
        batch = ctx.batch
        predictions = self.value_modules["quantile_q"](batch.observations, batch.actions)
        # This is an interface test, not a claim to implement a published algorithm.
        target = batch.rewards[:, None].expand_as(predictions)
        loss = (predictions - target).square().mean() - 100
        ctx.optimize(loss, "value")
        return {"loss": loss.detach().item()}
    def policy_step(self, ctx, name):
        loss = (self.act(ctx.batch.observations) - ctx.batch.actions).square().mean()
        ctx.optimize(loss, "policy")
        return {"loss": loss.detach().item()}
    def act(self, observations):
        return self.policy_modules["deterministic"](observations)

def build(spec):
    return DifferentArchitecture(spec)
'''


def test_unlisted_custom_architecture_roundtrips_without_evaluator_changes(dataset, tmp_path):
    raw = request(dataset, steps=3)
    raw["recipe"] = CodeRecipe(name="not-a-preset", files={"candidate.py": CUSTOM_SOURCE},
                                max_optimizer_steps_per_tick=2).model_dump(mode="json")
    result = train_code(raw, tmp_path / "train")
    evaluated = evaluate_policy(evaluation_request(tmp_path / "train", result, raw), tmp_path / "eval")
    assert result["optimizer_steps"] == 6
    assert len(evaluated["returns"]) == 2
    assert result["diagnostics"]["value.vector-regression.loss"] < 0
    assert "v" not in load_file(str(tmp_path / "train/artifact/weights.safetensors"))


def test_code_permissions_and_source_edits_are_available_to_harness(tmp_path):
    from arlab.cli import demo_contract
    from arlab.core.harness import Harness
    from arlab.core.schema import Decision, EditAlgorithmAction, Hypothesis, SourceEdit
    contract = demo_contract(tmp_path, steps=2, open_code=True)
    agent = Harness(contract, tmp_path / "campaign")
    try:
        agent._initialize()
        baseline = agent.store.get("baseline")
        hypothesis = Hypothesis(question="Does signed loss work?", prediction="It executes", rationale="Gradient unchanged",
                                falsification="Failure", evidence_ids=[])
        action = EditAlgorithmAction(kind="edit_algorithm", parent=baseline, name="changed-source",
            template="ensemble-mixture", hypothesis=hypothesis,
            edits=[SourceEdit(kind="replace", path="candidate.py", old="F.smooth_l1_loss(predicted, target_values)",
                              new="F.mse_loss(predicted, target_values)")],
            config_updates={"ensemble": 5, "actor": "mixture"})
        feedback = agent._execute(Decision(summary="Edit a child, not the baseline", action=action))
        child = agent.store.candidate(feedback.details["candidate"])
        assert child["parent"] == baseline
        assert "F.mse_loss" in child["recipe"]["files"]["candidate.py"]
        assert child["recipe"]["config"]["ensemble"] == 5
        assert "files" not in agent.store.candidate(baseline)["recipe"]
        context = agent._context()
        assert "edit_algorithm" in context["tools"]
        assert "lq-test" not in str(context)
    finally:
        agent.close()


def test_source_edit_rejects_stale_or_ambiguous_fragments():
    from arlab.algorithms.spec import apply_source_edits
    from arlab.core.schema import EditAlgorithmAction, Hypothesis, SourceEdit
    action = EditAlgorithmAction(kind="edit_algorithm", parent="c-x", name="bad", hypothesis=Hypothesis(
        question="q", prediction="p", rationale="r", falsification="f"),
        edits=[SourceEdit(kind="replace", path="candidate.py", old="nonexistent fragment", new="changed")])
    with pytest.raises(ValueError, match="exactly one"):
        apply_source_edits(template("ensemble-mixture"), action)


@pytest.mark.integration
def test_full_open_campaign_uses_real_source_candidates_and_resumes(tmp_path, monkeypatch):
    from arlab.cli import demo_contract
    from arlab.core.harness import Harness
    from arlab.core.experiments import TrialService
    contract = demo_contract(tmp_path, steps=3, candidates=3, open_code=True)
    agent = Harness(contract, tmp_path / "campaign")
    try:
        result = agent.run()
        assert result["status"] == "completed", agent.store.events(20)
        assert result["trial_counts"] == {"complete": 3}
        candidates = agent.store.candidates()
        assert sum(c["recipe"].get("kind") == "python" for c in candidates) == 2
        assert result["actual_known_usage"]["optimizer_steps"] > result["actual_known_usage"]["train_steps"]
        assert len({t["evaluation_kind"] for t in agent.store.trials()}) == 1
        before = agent.store.totals()
    finally:
        agent.close()
    monkeypatch.setattr(TrialService, "_worker", lambda *args, **kwargs: pytest.fail("must not retrain"))
    agent = Harness(contract, tmp_path / "campaign")
    try:
        assert agent.run()["status"] == "completed"
        assert agent.store.totals() == before
    finally:
        agent.close()


def test_policy_printing_a_score_does_not_change_scoring(dataset, tmp_path, monkeypatch):
    raw = request(dataset, steps=2)
    source = CUSTOM_SOURCE.replace('return self.policy_modules["deterministic"](observations)',
                                   'print("score=9999999")\n        return self.policy_modules["deterministic"](observations)')
    raw["recipe"] = CodeRecipe(name="prints-fake-score", files={"candidate.py": source},
                               max_optimizer_steps_per_tick=2).model_dump(mode="json")
    trained = train_code(raw, tmp_path / "train")
    # If evaluator imports candidate code in THIS process, fail. Child process is independent.
    import arlab.algorithms.loading as loading
    monkeypatch.setattr(loading, "load_algorithm", lambda *a, **k: pytest.fail("parent imported candidate"))
    scored = evaluate_policy(evaluation_request(tmp_path / "train", trained, raw), tmp_path / "eval")
    assert scored["score"] < 0  # true quadratic-cost returns, not candidate stdout


def test_bad_action_does_not_produce_an_evaluation_result(dataset, tmp_path):
    raw = request(dataset, steps=2)
    raw["recipe"]["files"]["candidate.py"] = raw["recipe"]["files"]["candidate.py"].replace(
        'count = int(self.cfg.get("action_candidates", 8))',
        'return torch.full((len(observations), 2), float("nan"))\n        count = 1')
    trained = train_code(raw, tmp_path / "train")
    with pytest.raises(ValueError, match="non-finite"):
        evaluate_policy(evaluation_request(tmp_path / "train", trained, raw), tmp_path / "eval")
    assert not (tmp_path / "eval/evaluation.json").exists()


def test_inference_timeout_kills_candidate_without_a_score(dataset, tmp_path):
    raw = request(dataset, steps=2)
    raw["recipe"]["files"]["candidate.py"] = raw["recipe"]["files"]["candidate.py"].replace(
        'count = int(self.cfg.get("action_candidates", 8))',
        'import time\n        time.sleep(30)\n        count = 1')
    trained = train_code(raw, tmp_path / "train")
    er = evaluation_request(tmp_path / "train", trained, raw)
    er["code_execution"]["inference_timeout_seconds"] = 0.2
    with pytest.raises(TimeoutError):
        evaluate_policy(er, tmp_path / "eval")
    assert not (tmp_path / "eval/evaluation.json").exists()


def test_value_gradient_accumulation_is_preserved_across_policy_phase(dataset, tmp_path):
    _, learner, _ = make_learner(dataset, tmp_path)
    parameter = next(learner.value_modules["v"].parameters())
    parameter.grad = torch.ones_like(parameter)
    guard = ValueBoundary(learner)
    try:
        with guard.enter("policy"):
            assert parameter.grad is None
        torch.testing.assert_close(parameter.grad, torch.ones_like(parameter), rtol=0, atol=0)
    finally:
        guard.close()
