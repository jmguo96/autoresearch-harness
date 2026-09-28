"""Generic open-algorithm training: persistence/resources, not fixed IQL equations."""
from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file

from arlab.algorithms.context import UpdateContext
from arlab.algorithms.guard import ValueBoundary, registered_modules
from arlab.algorithms.loading import encode_state, load_algorithm
from arlab.algorithms.protocol import BuildSpec, Update
from arlab.algorithms.spec import CodeRecipe
from arlab.core.io import canonical, digest, directory_digest, file_digest, read_json, runtime_info, write_json
from arlab.core.checkpoints import training_binding, series_binding, schedule_horizon, checkpoint_source, validate_checkpoint
from arlab.core.schema import CodeExecutionPolicy
from arlab.tasks.iql.data import OfflineReplay, load_dataset


def authorize(recipe: CodeRecipe, policy: CodeExecutionPolicy) -> None:
    if policy.mode == "disabled":
        raise PermissionError("Python candidates require owner-authorized code_execution")
    if recipe.max_optimizer_steps_per_tick > policy.max_optimizer_steps_per_tick:
        raise ValueError("Candidate optimizer-step bound exceeds owner policy")


def algorithm_state(learner: Any) -> dict[str, Any]:
    return {"modules": {k: m.state_dict() for k, m in registered_modules(learner).items()},
            "optimizers": {k: o.state_dict() for k, o in learner.optimizers.items()},
            "schedulers": {k: s.state_dict() for k, s in learner.schedulers.items()},
            "extra": learner.extra_state_dict()}


def restore_algorithm(learner: Any, state: dict[str, Any]) -> None:
    for field, objects in (("modules", registered_modules(learner)),
                           ("optimizers", learner.optimizers), ("schedulers", learner.schedulers)):
        if objects.keys() != state[field].keys():
            raise ValueError(f"Checkpoint {field} registry differs")
        for name, obj in objects.items():
            obj.load_state_dict(state[field][name])
    learner.load_extra_state_dict(state["extra"])


def save_checkpoint(path: Path, *, learner: Any, replay: OfflineReplay, guard: ValueBoundary,
                    tick: int, binding: str, diagnostics: dict[str, float], series: str | None = None) -> None:
    numpy_state = np.random.get_state()
    state = {"algorithm": algorithm_state(learner), "tick": tick, "binding": binding,
             "replay_rng": replay.generator.get_state(), "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
             "python_rng": random.getstate(),
             "numpy_rng": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
             "query_counts": dict(guard.counts), "optimizer_steps": guard.optimizations,
             "diagnostics": diagnostics}
    if series is not None:
        state["series_sha256"] = series
    pending = path.with_suffix(".pending")
    torch.save(state, pending)
    os.replace(pending, path)


def export_algorithm(path: Path, learner: Any, recipe: CodeRecipe, spec: BuildSpec,
                     replay: OfflineReplay, tick: int, binding: str, audit: dict[str, Any]) -> str:
    path.mkdir(parents=True, exist_ok=True)
    recipe.materialize(path / "source")
    tensors = {f"modules.{name}.{key}": tensor.detach().cpu().contiguous().clone()
               for name, module in registered_modules(learner).items()
               for key, tensor in module.state_dict().items()}
    extra = encode_state(learner.extra_state_dict(), tensors)
    tensors.update(observation_mean=torch.as_tensor(replay.mean).clone(),
                   observation_std=torch.as_tensor(replay.std).clone(),
                   action_low=torch.as_tensor(replay.low).clone(), action_high=torch.as_tensor(replay.high).clone())
    if any(not torch.isfinite(t).all() for t in tensors.values()):
        raise ValueError("Non-finite policy export")
    temporary = path / ".weights.pending"
    save_file(tensors, str(temporary))
    os.replace(temporary, path / "weights.safetensors")
    write_json(path / "algorithm.json", recipe)
    write_json(path / "policy.json", {"schema": 2, "architecture": "python-algorithm-v1",
        "observation_dim": spec.observation_dim, "action_dim": spec.action_dim,
        "weights_sha256": file_digest(path / "weights.safetensors"),
        "recipe_sha256": digest(recipe), "source_sha256": recipe.source_sha256,
        "training_binding": binding, "updates": tick, "total_ticks": spec.total_ticks,
        "extra_state": extra, "value_learning_audit": audit})
    return directory_digest(path)


def train_code(request: dict[str, Any], output: Path) -> dict[str, Any]:
    policy = CodeExecutionPolicy.model_validate(request.get("code_execution", {}))
    recipe = CodeRecipe.model_validate(request["recipe"])
    authorize(recipe, policy)
    if policy.mode == "docker" and os.environ.get("ARLAB_CODE_CONTAINER") != "1":
        raise PermissionError("Container-required candidate cannot execute on the local host")
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    binding = training_binding(request)
    complete = output / "training.json"
    if complete.exists():
        done = read_json(complete)
        if done["request_sha256"] != binding or directory_digest(output / "artifact") != done["artifact_sha256"]:
            raise ValueError("Completed candidate training has a different request/artifact")
        return done
    seed, steps = int(request["seed"]), int(request["steps"])
    if steps < 1:
        raise ValueError("Training tick count must be positive")
    device = request.get("device", "cpu")
    torch.set_num_threads(int(request.get("threads", 1)))
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    arrays, metadata = load_dataset(Path(request["data"]), request.get("dataset_sha256", ""))
    replay = OfflineReplay(arrays, metadata, seed)
    horizon = schedule_horizon(request)
    spec = BuildSpec(arrays["observations"].shape[1], arrays["actions"].shape[1], horizon, device, recipe.config)
    recipe.materialize(output / "source")
    learner = load_algorithm(recipe, output / "source", spec)
    guard = ValueBoundary(learner, max_parameters=policy.max_parameters)
    tick, diagnostics = 0, {}
    path = output / "checkpoint.pt"
    try:
        restore = checkpoint_source(request, path)
        if restore:
            # Only authorized worker checkpoints, never loaded in the scoring process.
            restored_path, expected = restore
            state = torch.load(restored_path, map_location=device, weights_only=True)
            validate_checkpoint(state, request, expected, int(state["tick"]))
            restore_algorithm(learner, state["algorithm"])
            tick = int(state["tick"])
            replay.generator.set_state(state["replay_rng"].cpu())
            torch.set_rng_state(state["torch_rng"].cpu())
            if device.startswith("cuda") and state["cuda_rng"]:
                torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
            random.setstate(state["python_rng"])
            ns = state["numpy_rng"]
            np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
            guard.counts.update(state["query_counts"])
            guard.optimizations = state["optimizer_steps"]
            diagnostics = state["diagnostics"]
        resumed_at, resumed_optimizations = tick, guard.optimizations
        period = min(1000, max(1, steps // 10))
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            while tick < steps:
                guard.tick_remaining = 0
                with guard.enter("schedule"):
                    schedule = list(learner.schedule(tick))
                if not 1 <= len(schedule) <= policy.max_optimizer_steps_per_tick * 4:
                    raise ValueError("Candidate update schedule exceeds phase budget or is empty")
                guard.tick_remaining = recipe.max_optimizer_steps_per_tick
                before = guard.optimizations
                diagnostics = {}
                for update in schedule:
                    if not isinstance(update, Update) or update.phase not in ("value", "policy"):
                        raise ValueError("Schedule must contain typed Update objects")
                    with guard.enter(update.phase):
                        context = UpdateContext(learner, guard, replay, tick=tick, device=device,
                                                remaining_optimizations=guard.tick_remaining)
                        fn = learner.value_step if update.phase == "value" else learner.policy_step
                        result = fn(context, update.name)
                        if not isinstance(result, dict):
                            raise ValueError("Algorithm updates return named scalar diagnostics")
                        for key, value in result.items():
                            value = float(value)
                            if not math.isfinite(value):
                                raise ValueError("Non-finite algorithm diagnostic")
                            diagnostics[f"{update.phase}.{update.name}.{key}"] = value
                if guard.optimizations == before:
                    raise ValueError("Training tick did not perform any registered optimization")
                tick += 1
                if tick % period == 0 or tick == steps:
                    stream.write(canonical({"tick": tick, "optimizer_steps": guard.optimizations, **diagnostics}) + "\n")
                    stream.flush()
                    save_checkpoint(path, learner=learner, replay=replay, guard=guard, tick=tick,
                                    binding=binding, diagnostics=diagnostics, series=series_binding(request))
        audit = guard.report()
        sha = export_algorithm(output / "artifact", learner, recipe, spec, replay, tick, binding, audit)
        if sum(p.stat().st_size for p in (output / "artifact").rglob("*") if p.is_file()) > policy.max_artifact_bytes:
            raise ValueError("Candidate export exceeds artifact budget")
        result = {"request_sha256": binding, "artifact_sha256": sha, "steps": tick,
                  "resumed_at": resumed_at, "schedule_steps": horizon,
                  "series_sha256": series_binding(request), "checkpoint_sha256": file_digest(path),
                  "train_seconds": time.monotonic() - started,
                  "diagnostics": diagnostics, "dataset_kind": metadata["kind"],
                  "optimizer_steps": guard.optimizations, "resumed_optimizer_steps": resumed_optimizations,
                  "value_learning_audit": audit, "recipe_sha256": digest(recipe),
                  "source_sha256": recipe.source_sha256, "runtime": runtime_info()}
        write_json(complete, result)
        return result
    finally:
        # Even failed candidates leave the observed boundary violations as diagnostics.
        write_json(output / "value-boundary.json", guard.report())
        guard.close()
