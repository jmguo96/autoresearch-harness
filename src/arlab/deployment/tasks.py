"""Post-install tasks. This module may import the ML runtime; bootstrap may not."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from arlab.core.io import file_digest, read_json, runtime_info, source_digest, write_json
from arlab.core.presets import research_budget, adaptive_exploration
from arlab.core.schema import Budget, Case, CodeExecutionPolicy, ModelConfig, ResearchContract
from arlab.deployment.common import atomic_json, exclusive_lock
from arlab.tasks.iql.data import import_hdf5, load_dataset
from arlab.tasks.iql.environments import D4RL_IDS, make_environment, reference
from arlab.tasks.iql.prepare import download_raw
from arlab.tasks.iql.recipe import IQLRecipe

CONVERSION = "d4rl-v2-qlearning-explicit-timeouts-drop-with-links-v2"


def official_filename(env_id: str) -> str:
    reference(env_id)
    env, quality = env_id.removesuffix("-v2").split("-", 1)
    return f"{env}_{quality.replace('-', '_')}-v2.hdf5"


def prepare_one(env_id: str, output: Path, *, raw: Path | None = None,
                expected_sha256: str = "", offline: bool = False) -> dict[str, Any]:
    """Reuse only a validated conversion with the same raw hash and converter.

    A previously recorded conversion hash must still match. Corrupt files are
    reported rather than silently replaced with another dataset or synthetic data.
    """
    _, _, obs_dim, action_dim = reference(env_id)
    if expected_sha256:
        import re
        if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
            raise ValueError("Raw SHA256 must contain 64 hexadecimal characters")
        expected_sha256 = expected_sha256.lower()
    output = output.absolute()
    if output.is_symlink():
        raise ValueError("Dataset destination cannot be a symlink")
    output.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(output.with_name(output.name + ".prepare.lock")):
        receipt_path = output.with_name(output.name + ".preparation.json")
        if receipt_path.is_symlink():
            raise ValueError("Dataset receipt cannot be a symlink")
        previous = read_json(receipt_path) if receipt_path.exists() else None
        raw = raw or output.with_suffix(".raw.hdf5")
        if not raw.exists():
            if offline:
                raise FileNotFoundError(f"Offline mode needs raw data: {raw}")
            if raw != output.with_suffix(".raw.hdf5"):
                raise FileNotFoundError(f"Explicit raw data path does not exist: {raw}")
            download_raw(env_id, raw, expected_sha256)
        if raw.is_symlink() or not raw.is_file():
            raise ValueError("Raw dataset must be a regular non-symlink file")
        raw_hash = file_digest(raw)
        if expected_sha256 and raw_hash != expected_sha256:
            raise ValueError("Raw dataset checksum mismatch")
        existed = output.exists()
        if not existed:
            import h5py
            with h5py.File(raw, "r") as h5:
                if h5["observations"].ndim != 2 or h5["observations"].shape[1] != obs_dim:
                    raise ValueError("Dataset dimensions disagree with the frozen D4RL task")
            import_hdf5(raw, output, env_id, action_dim=action_dim)
        arrays, metadata = load_dataset(output)
        if (metadata.get("environment") != env_id or metadata.get("raw_sha256") != raw_hash or
                metadata.get("conversion") != CONVERSION or "next_indices" not in arrays):
            raise ValueError("Existing conversion has different data/environment/semantics; use a new output path")
        if arrays["observations"].shape[1] != obs_dim or arrays["actions"].shape[1] != action_dim:
            raise ValueError("Dataset dimensions disagree with the frozen D4RL task")
        digest = file_digest(output)
        if previous and (previous["sha256"] != digest or previous["raw_sha256"] != raw_hash):
            raise ValueError("Dataset changed after the recorded conversion")
        result = {"environment": env_id, "path": str(output), "raw_path": str(raw.resolve()),
                  "sha256": digest, "raw_sha256": raw_hash, "conversion": CONVERSION,
                  "transitions": len(arrays["rewards"]), "observation_dim": obs_dim, "action_dim": action_dim,
                  "raw_identity": "trusted_hash_matched" if expected_sha256 else "local_hash_not_independent_authentication"}
        atomic_json(receipt_path, result)
        return {**result, "reused": existed}


def prepare_many(root: Path, envs: list[str], *, raw_dir: Path | None = None,
                 checksums: dict[str, str] | None = None, offline: bool = False) -> dict[str, Any]:
    if not envs or len(set(envs)) != len(envs) or any(env not in D4RL_IDS for env in envs):
        raise ValueError("Expected a nonempty set of known D4RL-v2 environments")
    checksums = checksums or {}
    if set(checksums) - set(D4RL_IDS) or not all(isinstance(x, str) for x in checksums.values()):
        raise ValueError("Checksum manifest must map known task names to SHA256 strings")
    report: dict[str, Any] = {"schema": 1, "requested": envs, "status": "running", "datasets": []}
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(root / ".datasets.lock"):
        for env in envs:
            try:
                raw = None
                if raw_dir:
                    matches = [p for p in (raw_dir / f"{env}.raw.hdf5", raw_dir / official_filename(env),
                                           raw_dir / f"{env}.hdf5") if p.is_file()]
                    if not matches:
                        raise FileNotFoundError(f"No raw file for {env} in {raw_dir}")
                    raw = matches[0]
                result = prepare_one(env, root / f"{env}.npz", raw=raw,
                                     expected_sha256=checksums.get(env, ""), offline=offline)
                report["datasets"].append(result)
                atomic_json(root / "datasets.json", report)
                print(f"Prepared {env}: {result['transitions']} transitions", flush=True)
            except Exception as exc:
                report.update(status="failed", failed_environment=env,
                              error=f"{type(exc).__name__}: {exc}")
                atomic_json(root / "datasets.json", report)
                raise
        report["status"] = "ready"
        atomic_json(root / "datasets.json", report)
    return report


def make_contract(settings: dict[str, Any]) -> ResearchContract:
    root = Path(settings["root"])
    if settings["profile"] == "smoke":
        from arlab.cli import demo_contract
        raw = demo_contract(root / "fixture", steps=settings["smoke_steps"], candidates=3).model_dump(mode="json")
    else:
        cases = []
        for env in settings["envs"]:
            path = root / "data" / f"{env}.npz"
            arrays, metadata = load_dataset(path)
            _, _, obs_dim, action_dim = reference(env)
            if (metadata.get("environment") != env or arrays["observations"].shape[1] != obs_dim or
                    arrays["actions"].shape[1] != action_dim):
                raise ValueError("Cannot create contract from mismatched data")
            split = "test" if settings["suite12"] and "medium-expert" in env else (
                "validation" if settings["suite12"] and "medium-replay" in env else "dev")
            cases.append(Case(name=env, split=split, data=str(path.resolve()),
                              environment={"kind": "d4rl", "id": env}, dataset_sha256=file_digest(path)))
        raw = ResearchContract(name="d4rl-iql-insample", goal="Research open in-sample value-learning algorithms; compare with the unchanged IQL baseline.",
            cases=cases, seed_recipe=IQLRecipe().model_dump(mode="json"), train_seeds=[0, 1, 2],
            eval_seeds=list(range(1000, 1010)), rungs=[5000, 20000, 50000, 100000, 250000, 1000000], max_candidates=128,
            max_revisions=10000, exploration=adaptive_exploration(), budget=research_budget()).model_dump(mode="json")
    raw["device"] = settings["device"]
    raw["model"] = ModelConfig(provider=settings["provider"], model=settings["model"],
                                base_url=settings["base_url"]).model_dump(mode="json")
    raw["code_execution"] = CodeExecutionPolicy(mode=settings["code_mode"], docker_image=settings["image"],
                                              container_gpu=settings["device"] == "cuda").model_dump(mode="json")
    raw["research_access"]["enabled"] = bool(settings.get("research_online", False))
    raw["research_access"]["allowed_hosts"] = list(dict.fromkeys(
        raw["research_access"]["allowed_hosts"] + settings.get("research_hosts", [])))
    if settings.get("suite9"):
        from arlab.tasks.iql.study import apply_nine_task_protocol
        raw = apply_nine_task_protocol(raw).model_dump(mode="json")
        if not settings["model"] or "REPLACE" in settings["model"]:
            from arlab.classroom.kimi import default_model_settings
            raw["model"] = ModelConfig.model_validate(default_model_settings()).model_dump(mode="json")
        raw["research_access"]["enabled"] = bool(settings.get("research_online", False))
    return ResearchContract.model_validate(raw)


def configure(root: Path) -> None:
    contract = make_contract(read_json(root / "settings.json"))
    path = root / "configs/research.json"
    if path.exists():
        existing = ResearchContract.model_validate(read_json(path))
        old = [(c.name, c.split, c.data, c.dataset_sha256) for c in existing.cases]
        new = [(c.name, c.split, c.data, c.dataset_sha256) for c in contract.cases]
        if old != new or existing.device != contract.device or existing.code_execution != contract.code_execution or existing.final_evaluation != contract.final_evaluation:
            write_json(path.with_name("research.proposed.json"), contract)
            raise ValueError("Existing research config conflicts with deployment; saved research.proposed.json, did not overwrite")
        # Preserve deliberate owner edits to budgets and model configuration.
    else:
        write_json(path, contract)
    write_json(root / "configs/iql.baseline.json", IQLRecipe())


def check_environment(env_id: str) -> dict[str, Any]:
    import numpy as np
    env = make_environment({"kind": "d4rl", "id": env_id})
    try:
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        result = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        if len(result) not in (4, 5) or not np.isfinite(result[0]).all() or not np.isfinite(result[1]):
            raise ValueError("Environment step returned invalid observation/reward")
        return {"environment": env_id, "reset_and_step": True, "observation_dim": len(obs),
                "action_dim": env.action_space.shape[0], "normalization_checked": True}
    finally:
        env.close()


def check(root: Path) -> None:
    import torch
    settings = read_json(root / "settings.json")
    device = settings["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested/detected but PyTorch cannot use it; choose --device cpu in a new root or repair the driver")
    value = torch.ones(4, device=device).sum().item()
    if value != 4:
        raise RuntimeError("Device computation failed")
    result = {"status": "running", "runtime": runtime_info(), "source_sha256": source_digest(),
              "device": device, "device_tensor_computation": True, "environments": []}
    if settings["profile"] == "d4rl":
        for env in settings["envs"]:
            result["environments"].append(check_environment(env))
            atomic_json(root / "checks.json", result)
    result["status"] = "passed"
    atomic_json(root / "checks.json", result)


def _arlab(argv: list[str]) -> None:
    from arlab.core.jobs import terminate_group
    command = [sys.executable, "-m", "arlab", *argv]
    process = subprocess.Popen(command, start_new_session=True)
    try:
        code = process.wait()
    except BaseException:
        # Let the CLI unwind its worker/policy cleanup before escalating.
        terminate_group(process, grace=8)
        raise
    if code:
        raise subprocess.CalledProcessError(code, command)


def smoke(root: Path, run_id: str) -> None:
    settings = read_json(root / "settings.json")
    target = root / "checks" / run_id
    target.mkdir(parents=True, exist_ok=False)
    steps = settings["smoke_steps"]
    result: dict[str, Any] = {"purpose": "installation_smoke_not_algorithm_benchmark", "status": "running",
                              "profile": settings["profile"], "training_steps": steps, "llm_calls": 0}
    if settings["profile"] == "smoke":
        # Only bundled, reviewed templates; no remote model-generated code.
        from arlab.cli import _run, demo_contract
        raw = demo_contract(target / "open", steps=steps, candidates=3, open_code=True).model_dump(mode="json")
        raw["device"] = settings["device"]
        if settings["code_mode"] != "disabled":
            raw["code_execution"] = CodeExecutionPolicy(mode=settings["code_mode"],
                docker_image=settings["image"], container_gpu=settings["device"] == "cuda").model_dump(mode="json")
        report = _run(ResearchContract.model_validate(raw), target / "open/campaign")
        if report["status"] != "completed":
            raise RuntimeError("Synthetic pipeline did not complete")
        result["campaign"] = str(target / "open/campaign/report.json")
    else:
        env = settings["envs"][0]
        data = root / "data" / f"{env}.npz"
        _arlab(["train", "--dataset", str(data), "--recipe", str(root / "configs/iql.baseline.json"),
                 "--output", str(target / "iql"), "--steps", str(steps), "--device", settings["device"]])
        _arlab(["evaluate", "--policy", str(target / "iql/artifact"), "--env", env,
                 "--output", str(target / "iql-eval"), "--episodes", "1"])
        result["baseline"] = {"training": str(target / "iql/training.json"),
                              "evaluation": str(target / "iql-eval/evaluation.json")}
        if settings["code_mode"] != "disabled":
            from arlab.algorithms.templates import template
            write_json(target / "candidate.json", template("ensemble-mixture", tiny=True))
            permission = ["--code-mode", settings["code_mode"]]
            if settings["image"]:
                permission += ["--code-image", settings["image"]]
            _arlab(["train", "--dataset", str(data), "--recipe", str(target / "candidate.json"),
                     "--output", str(target / "candidate"), "--steps", str(steps),
                     "--device", settings["device"], *permission])
            _arlab(["evaluate", "--policy", str(target / "candidate/artifact"), "--env", env,
                     "--output", str(target / "candidate-eval"), "--episodes", "1", *permission])
            result["open_code_candidate"] = "trained_and_evaluated"
    result["status"] = "passed"
    atomic_json(target / "smoke.json", result)
    atomic_json(root / "smoke-latest.json", result)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("datasets")
    d.add_argument("--data-root", type=Path, required=True)
    d.add_argument("--envs", nargs="+", required=True)
    d.add_argument("--raw-dir", type=Path)
    d.add_argument("--checksums", type=Path)
    d.add_argument("--offline", action="store_true")
    for cmd in ("configure", "check", "smoke"):
        q = sub.add_parser(cmd)
        q.add_argument("--root", type=Path, required=True)
        if cmd == "smoke":
            q.add_argument("--run-id", required=True)
    args = p.parse_args(argv)
    if args.command == "datasets":
        prepare_many(args.data_root, args.envs, raw_dir=args.raw_dir, offline=args.offline,
                     checksums=read_json(args.checksums) if args.checksums else None)
    elif args.command == "configure":
        configure(args.root)
    elif args.command == "check":
        check(args.root)
    elif args.command == "smoke":
        if not args.run_id.replace("-", "").isalnum():
            raise ValueError("Invalid smoke run ID")
        smoke(args.root, args.run_id)
    return 0


if __name__ == "__main__":
    import signal

    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    raise SystemExit(main())
