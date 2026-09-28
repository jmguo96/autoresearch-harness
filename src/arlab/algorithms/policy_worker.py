"""Candidate-side inference. Receives observations, never environment rewards or scores."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

from arlab.algorithms.guard import ValueBoundary, registered_modules
from arlab.algorithms.loading import decode_state, load_algorithm
from arlab.algorithms.policy import validate_artifact
from arlab.algorithms.protocol import BuildSpec
from arlab.core.io import canonical, source_digest
from arlab.core.schema import CodeExecutionPolicy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--expected-hash", required=True)
    parser.add_argument("--binding", required=True)
    parser.add_argument("--max-parameters", required=True, type=int)
    parser.add_argument("--max-artifact-bytes", required=True, type=int)
    parser.add_argument("--implementation-sha256", required=True)
    args = parser.parse_args()
    protocol = sys.stdout
    sys.stdout = sys.stderr  # ordinary candidate print() cannot corrupt the JSON channel
    def send(payload):
        protocol.write(canonical(payload) + "\n")
        protocol.flush()
    guard = None
    try:
        if source_digest() != args.implementation_sha256:
            raise ValueError("Policy worker implementation differs from evaluator")
        torch.set_num_threads(1)
        torch.manual_seed(0)
        manifest, recipe, tensors = validate_artifact(args.artifact, args.expected_hash, args.binding,
                                                     CodeExecutionPolicy(mode="trusted_local", max_artifact_bytes=args.max_artifact_bytes))
        spec = BuildSpec(manifest.observation_dim, manifest.action_dim, manifest.total_ticks,
                         "cpu", recipe.config, inference=True)
        learner = load_algorithm(recipe, args.artifact / "source", spec)
        modules = registered_modules(learner)
        expected = {f"modules.{name}.{key}" for name, m in modules.items() for key in m.state_dict()}
        present = {key for key in tensors if key.startswith("modules.")}
        if expected != present:
            raise ValueError("Policy module/weight registry differs from export")
        for name, module in modules.items():
            prefix = f"modules.{name}."
            module.load_state_dict({k[len(prefix):]: v for k, v in tensors.items() if k.startswith(prefix)}, strict=True)
            module.eval()
        learner.load_extra_state_dict(decode_state(manifest.extra_state, tensors))
        guard = ValueBoundary(learner, max_parameters=args.max_parameters)
        send({"status": "ready"})
        for line in sys.stdin:
            if len(line) > 1_000_000:
                raise ValueError("Policy input exceeds byte budget")
            request = json.loads(line)
            if request.get("kind") == "reset":
                seed = int(request["seed"])
                torch.manual_seed(seed)
                random.seed(seed)
                np.random.seed(seed % 2**32)
                if hasattr(learner, "reset"):
                    with guard.enter("inference"):
                        learner.reset(seed)
                send({"status": "ready"})
            elif request.get("kind") == "act":
                observation = torch.as_tensor(request["observation"], dtype=torch.float32).unsqueeze(0)
                with guard.enter("inference"):
                    action = learner.act(observation)
                if not isinstance(action, torch.Tensor) or action.shape != (1, manifest.action_dim):
                    raise ValueError("act() must return [batch, action_dim] tensor")
                if not torch.isfinite(action).all():
                    raise ValueError("act() returned non-finite actions")
                send({"action": action.detach().cpu().squeeze(0).tolist(), "audit": guard.report()})
            else:
                raise ValueError("Unknown policy request")
        return 0
    except Exception as exc:
        send({"error": f"{type(exc).__name__}: {exc}"[:4000]})
        return 1
    finally:
        if guard is not None:
            guard.close()


if __name__ == "__main__":
    raise SystemExit(main())
