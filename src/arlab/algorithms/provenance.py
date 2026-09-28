"""Best-effort tensor lineage for policy-only action-value estimates.

Ordinary PyTorch operations, including detach(), preserve this Tensor subclass.
It catches accidental cached generated-action Q values reused in a value loss.
Converting through Python numbers/NumPy or deliberately stripping the subclass
can erase lineage: this is additional diagnostics, NOT a general information-flow
proof. Candidates must honestly register value roots and use the checked API.
"""
from __future__ import annotations

from typing import Any

import torch


class PolicyOnlyValue(torch.Tensor):
    """A value estimated on non-recorded actions, permitted only for policy learning."""


def mark_policy_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.as_subclass(PolicyOnlyValue)
    if isinstance(value, tuple):
        return tuple(mark_policy_value(v) for v in value)
    if isinstance(value, list):
        return [mark_policy_value(v) for v in value]
    if isinstance(value, dict):
        return {k: mark_policy_value(v) for k, v in value.items()}
    return value
