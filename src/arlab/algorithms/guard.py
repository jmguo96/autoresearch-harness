"""Observable in-sample invariants, not a proof about arbitrary Python semantics.

All action-value roots must be declared. During value learning their first two
arguments must match a witnessed, paired dataset batch. During policy extraction
novel actions are allowed but value parameters AND buffers are read-only.

A dishonest plugin can hide a value function or defeat in-process hooks. Use code
review and OS isolation; never present this instrumentation as formal verification.
"""
from __future__ import annotations

import inspect
from collections import Counter
from contextlib import contextmanager
from typing import Any, Iterator

import torch
from torch import nn

from arlab.algorithms.protocol import Learner
from arlab.tasks.iql.data import Batch


class BoundaryViolation(ValueError):
    """An observed candidate action violated the frozen value-learning contract."""


def registered_modules(learner: Learner) -> dict[str, nn.Module]:
    values, policies = dict(learner.value_modules), dict(learner.policy_modules)
    if values.keys() & policies.keys() or not (values or policies):
        raise ValueError("Declare at least one module, with distinct names across ownership groups")
    modules = {**values, **policies}
    if any(not name.isidentifier() or not isinstance(m, nn.Module) for name, m in modules.items()):
        raise ValueError("Module names must be identifiers and values must be nn.Module instances")
    if not set(learner.action_value_names).issubset(values):
        raise ValueError("Declare every action-value root by a registered value-module name")
    return modules


def tensors_in(modules: dict[str, nn.Module]) -> dict[int, torch.Tensor]:
    return {id(t): t for m in modules.values() for t in (*m.parameters(), *m.buffers())}


class ValueBoundary:
    def __init__(self, learner: Learner, *, max_parameters: int = 50_000_000):
        self.learner = learner
        self.modules = registered_modules(learner)
        params = {id(p): p for m in self.modules.values() for p in m.parameters()}
        if sum(p.numel() for p in params.values()) > max_parameters:
            raise ValueError("Candidate exceeds the owner-authorized parameter budget")
        self._registered_ids = {name: id(m) for name, m in self.modules.items()}
        self._tensor_ids = set(tensors_in(self.modules))
        self.phase = "idle"
        self.witnesses: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.counts: Counter[str] = Counter()
        self.optimizations = 0
        self.handles = []
        self.tick_remaining = 0
        self._optimizer_ids = {name: id(opt) for name, opt in learner.optimizers.items()}
        for optimizer in learner.optimizers.values():
            self.handles.append(optimizer.register_step_pre_hook(self._optimizer_step))
        for name in learner.action_value_names:
            module = self.modules[name]
            parameters = list(inspect.signature(module.forward).parameters)
            if len(parameters) < 2:
                raise ValueError("Action-value forward must accept observations and actions")
            def hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any],
                     names: list[str] = parameters, label: str = name) -> None:
                observation = args[0] if args else kwargs.get(names[0])
                action = args[1] if len(args) > 1 else kwargs.get(names[1])
                self._query(label, observation, action)
            self.handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))
            def post_hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any,
                          names: list[str] = parameters) -> Any:
                observation = args[0] if args else kwargs.get(names[0])
                action = args[1] if len(args) > 1 else kwargs.get(names[1])
                if self.phase in ("policy", "inference") and not self.is_recorded(observation, action):
                    from arlab.algorithms.provenance import mark_policy_value
                    self.counts[f"{self.phase}_generated_q_queries"] += 1
                    return mark_policy_value(output)
                return output
            self.handles.append(module.register_forward_hook(post_hook, with_kwargs=True))

    def witness(self, batch: Batch) -> None:
        # Clone: mutating a candidate's tensor cannot rewrite its provenance witness.
        if batch.indices is None:
            raise BoundaryViolation("Value batch has no recorded row identities")
        self.witnesses.append((batch.observations.detach().clone(), batch.actions.detach().clone()))

    def is_recorded(self, observations: Any, actions: Any) -> bool:
        return (isinstance(observations, torch.Tensor) and isinstance(actions, torch.Tensor)
                and any(observations.shape == s.shape and actions.shape == a.shape
                        and torch.equal(observations, s) and torch.equal(actions, a)
                        for s, a in self.witnesses))

    def _query(self, name: str, observations: Any, actions: Any) -> None:
        if not isinstance(observations, torch.Tensor) or not isinstance(actions, torch.Tensor):
            raise BoundaryViolation(f"Action-value root {name} requires tensor inputs")
        if self.phase == "value":
            match = self.is_recorded(observations, actions)
            if not match:
                self.counts["rejected_value_queries"] += 1
                raise BoundaryViolation("Q/V learning queried an unpaired or unrecorded state-action batch")
            self.counts["recorded_value_queries"] += 1
        elif self.phase in ("policy", "inference"):
            # Deliberately allowed, even when actions have actor/action gradients.
            self.counts[f"{self.phase}_action_value_queries"] += 1
        else:
            raise BoundaryViolation("Action-value queries require an explicit update/inference phase")

    def _optimizer_step(self, optimizer: torch.optim.Optimizer, args: Any, kwargs: Any) -> None:
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        if not parameters or any(id(p) not in self.allowed_parameter_ids() for p in parameters):
            raise BoundaryViolation("Optimizer touches parameters owned by another phase")
        if self.tick_remaining <= 0:
            raise BoundaryViolation("Candidate exceeded its optimizer-step reservation")
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
            raise ValueError("Non-finite gradient in registered optimizer")
        self.tick_remaining -= 1
        self.optimizations += 1

    def check_registry(self) -> None:
        modules = registered_modules(self.learner)
        if {name: id(opt) for name, opt in self.learner.optimizers.items()} != self._optimizer_ids:
            raise BoundaryViolation("Optimizer registry changed after initialization")
        if ({name: id(m) for name, m in modules.items()} != self._registered_ids
                or set(tensors_in(modules)) != self._tensor_ids):
            raise BoundaryViolation("Replace architecture between candidates, not behind the active registry")

    @contextmanager
    def enter(self, phase: str) -> Iterator[None]:
        if self.phase != "idle" or phase not in ("value", "policy", "inference", "schedule"):
            raise BoundaryViolation("Invalid or nested algorithm phase")
        self.check_registry()
        values = tensors_in(dict(self.learner.value_modules))
        policies = tensors_in(dict(self.learner.policy_modules))
        # Shared encoders are allowed. Shared tensors belong to the value side;
        # policy optimization cannot modify them, but value updates may.
        protected = ({k: v for k, v in policies.items() if k not in values} if phase == "value"
                     else values if phase == "policy" else {**values, **policies})
        snapshots = {k: v.detach().clone() for k, v in protected.items()}
        all_parameters = {id(p): p for m in self.modules.values() for p in m.parameters()}
        flags = {k: p.requires_grad for k, p in all_parameters.items()}
        saved_grads = {k: None if p.grad is None else p.grad.detach().clone()
                       for k, p in all_parameters.items() if k in protected}
        all_submodules = {id(m): m for root in self.modules.values() for m in root.modules()}
        modes = {k: m.training for k, m in all_submodules.items()}
        protected_roots = (self.learner.policy_modules if phase == "value" else
                           self.learner.value_modules if phase == "policy" else self.modules)
        for m in protected_roots.values():
            m.eval()
        # Restore original training modes of shared value encoders during value learning.
        if phase == "value":
            for root in self.learner.value_modules.values():
                for m in root.modules():
                    m.training = modes[id(m)]
        for k, p in all_parameters.items():
            if k in protected:
                p.requires_grad_(False)
                p.grad = None
        self.phase, self.witnesses = phase, []
        try:
            yield
        finally:
            changed = [k for k, tensor in protected.items()
                       if tensor.shape != snapshots[k].shape or not torch.equal(tensor.detach(), snapshots[k])]
            with torch.no_grad():
                for k in changed:
                    if protected[k].shape == snapshots[k].shape:
                        protected[k].copy_(snapshots[k])
            crossed_gradients = [k for k in saved_grads if all_parameters[k].grad is not None]
            for k, p in all_parameters.items():
                p.requires_grad_(flags[k])
                if k in saved_grads:
                    p.grad = saved_grads[k]
            for k, m in all_submodules.items():
                m.training = modes[k]
            self.phase, self.witnesses = "idle", []
            self.check_registry()
            if changed or crossed_gradients:
                self.counts["rejected_cross_phase_mutations"] += 1
                raise BoundaryViolation("A phase modified protected parameters or buffers (restored when possible)")

    def allowed_parameter_ids(self) -> set[int]:
        value_ids = {id(p) for m in self.learner.value_modules.values() for p in m.parameters()}
        policy_ids = {id(p) for m in self.learner.policy_modules.values() for p in m.parameters()}
        return value_ids if self.phase == "value" else policy_ids - value_ids if self.phase == "policy" else set()

    def report(self) -> dict[str, Any]:
        return {"mode": "registered_runtime_checks", "value_actions": "recorded_pairs",
                "generated_actions_allowed_in_policy": True,
                "formal_verification": False, "requires_honest_complete_registration": True,
                "optimizer_steps": self.optimizations, **dict(self.counts)}

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
