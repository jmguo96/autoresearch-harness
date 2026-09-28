"""Training identity separates a fixed trajectory from observation endpoints.

The host validates hashes; only the training worker deserializes checkpoint
state. Continuation never edits an earlier snapshot or changes the algorithm,
seed, data, schedule horizon or execution policy. No weights enter the evaluator
via pickle. A runtime failure may conservatively consume a reserved segment.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from arlab.core.io import digest, file_digest


# 三个身份不要混淆：分支组织假设；候选保存源码；series 保存一条训练轨迹。
# 端点请求保留 steps，只有 resume_from 这个执行来路不参与端点绑定。
def training_binding(request: dict[str, Any]) -> str:
    """Execution source is provenance, not a change to the scientific endpoint."""
    return digest({k: v for k, v in request.items() if k != "resume_from"})


def schedule_horizon(request: dict[str, Any]) -> int:
    value = int(request.get("schedule_steps", request["steps"]))
    if value < int(request["steps"]):
        raise ValueError("Stop point exceeds the frozen schedule horizon")
    return value


# 轨迹身份移除暂停端点，但保留完整训练日程；换源码、数据或 seed 就不是续训。
def series_binding(request: dict[str, Any]) -> str:
    fields = {k: v for k, v in request.items() if k not in ("steps", "resume_from", "schedule_steps")}
    fields["schedule_steps"] = schedule_horizon(request)
    return digest(fields)


# 优先恢复同一请求自己的检查点，其次才使用经过哈希核验的较早端点。
def checkpoint_source(request: dict[str, Any], own: Path) -> tuple[Path, str] | None:
    """Choose a same-request recovery or a verified earlier endpoint."""
    if own.exists():
        if own.is_symlink():
            raise ValueError("Checkpoint cannot be a symlink")
        return own, training_binding(request)
    source = request.get("resume_from")
    if not source:
        return None
    path = Path(source["path"])
    if path.is_symlink() or not path.is_file() or file_digest(path) != source["sha256"]:
        raise ValueError("Continuation checkpoint missing or changed")
    if source["series_sha256"] != series_binding(request):
        raise ValueError("Continuation changes data, seed, algorithm or schedule")
    if not 0 < int(source["steps"]) < int(request["steps"]):
        raise ValueError("Continuation needs a strictly earlier checkpoint")
    return path, source["request_sha256"]


def validate_checkpoint(state: dict, request: dict, expected_binding: str, step: int) -> None:
    if state["binding"] != expected_binding:
        raise ValueError("Checkpoint request binding mismatch")
    # Old standalone requests remain readable within the same invocation only.
    if "series_sha256" in state and state["series_sha256"] != series_binding(request):
        raise ValueError("Checkpoint training-series mismatch")
    if request.get("resume_from") and "series_sha256" not in state:
        raise ValueError("Legacy checkpoints have no verified continuation identity")
    if not 0 <= step <= int(request["steps"]):
        raise ValueError("Checkpoint step is outside requested interval")
    if expected_binding != training_binding(request) and step != int(request["resume_from"]["steps"]):
        raise ValueError("Continuation checkpoint step differs from its receipt")
