"""Small structural guard against relabeling scalar tuning as a new IQL method."""
from __future__ import annotations

from typing import Any


def _mechanism(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _mechanism(item) for key, item in value.items() if key != "name"}
    if isinstance(value, list):
        return [_mechanism(item) for item in value]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "<number>"
    return value


def method_changed(parent: dict[str, Any], child: dict[str, Any]) -> bool:
    """Reject scalar-only recipe changes; self-review handles semantic judgment.

    This is deliberately conservative, not a novelty detector. Python source
    changes remain visible as text; a reviewer must still assess their substance.
    """
    return _mechanism(parent) != _mechanism(child)
