"""Statistics preserve the task / training seed hierarchy, not just rollout count."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np


def interval(values: list[float], *, seed: int = 0, draws: int = 2000) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("Need finite observations")
    estimate = float(x.mean())
    if len(x) < 2:
        return {"mean": estimate, "lower": None, "upper": None, "n": len(x)}
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(draws, len(x)))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"mean": estimate, "lower": float(lo), "upper": float(hi), "n": len(x)}


def iqm(values: list[float]) -> float:
    """Exact fractional trimming, including small arrays (no scipy dependency)."""
    x = np.sort(np.asarray(values, dtype=float))
    if len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("Need finite values")
    n = len(x)
    weights = np.maximum(0, np.minimum(np.arange(1, n + 1), 0.75 * n) - np.maximum(np.arange(n), 0.25 * n))
    return float(np.dot(x, weights) / (0.5 * n))


def summarize(trials: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[float]] = defaultdict(list)
    for trial in trials:
        groups[trial["case"]].append(float(trial["score"]))
    if not groups:
        return {"tasks": {}, "macro_mean": None, "iqm": None}
    means = [float(np.mean(v)) for v in groups.values()]
    return {"tasks": {k: interval(v) for k, v in groups.items()},
            "macro_mean": float(np.mean(means)), "iqm": iqm(means),
            "note": "Intervals resample independent training seeds, not episodes. IQM is over task means."}


def paired_comparison(baseline: list[dict[str, Any]], candidate: list[dict[str, Any]],
                      *, draws: int = 2000, expected_cells: set[tuple[str, int, int]] | None = None) -> dict[str, Any]:
    """Only compare identical (task, seed, steps, metric) cells; never best-of-test."""
    def key(t: dict[str, Any]) -> tuple[Any, ...]:
        return t["case"], t["train_seed"], t["steps"], t["metric"]
    b = {key(t): t["score"] for t in baseline}
    c = {key(t): t["score"] for t in candidate}
    shared = b.keys() & c.keys()
    observed_b = {k[:3] for k in b}
    observed_c = {k[:3] for k in c}
    expected = expected_cells if expected_cells is not None else observed_b | observed_c
    panel_complete = bool(expected) and observed_b == observed_c == expected and b.keys() == c.keys()
    coverage = {"expected": len(expected), "baseline": len(observed_b & expected),
                "candidate": len(observed_c & expected),
                "missing_baseline": [list(x) for x in sorted(expected - observed_b)],
                "missing_candidate": [list(x) for x in sorted(expected - observed_c)]}
    if not shared:
        return {"status": "insufficient_evidence", "pairs": 0, "complete": False,
                "contract_panel_complete": False, "coverage": coverage}
    groups: dict[str, list[float]] = defaultdict(list)
    for k in sorted(shared):
        groups[k[0]].append(c[k] - b[k])
    macro = float(np.mean([np.mean(g) for g in groups.values()]))
    enough = panel_complete and len(groups) >= 2 and all(len(g) >= 2 for g in groups.values())
    lo = hi = None
    if enough:
        rng = np.random.default_rng(0)
        samples = []
        arrays = [np.asarray(v) for v in groups.values()]
        for _ in range(draws):
            selected = rng.integers(0, len(arrays), len(arrays))
            samples.append(np.mean([rng.choice(arrays[i], len(arrays[i])).mean() for i in selected]))
        lo, hi = map(float, np.quantile(samples, [0.025, 0.975]))
    status = "insufficient_evidence" if not enough else ("positive_evidence" if lo > 0 else "inconclusive")
    return {"status": status, "pairs": len(shared), "complete": panel_complete, "contract_panel_complete": panel_complete if expected_cells is not None else None,
            "matched_observed_cells": b.keys() == c.keys(), "coverage": coverage,
            "macro_delta": macro, "lower": lo, "upper": hi,
            "per_task_delta": {k: float(np.mean(v)) for k, v in groups.items()},
            "warning": "Exploratory comparison; adaptive search and repeated tests require independent confirmation."}
