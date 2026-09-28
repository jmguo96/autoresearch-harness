from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from arlab.deployment.common import atomic_json, public_environment, read_json
from arlab.deployment.manager import Deployer, main, parser, plan, validate_options
from arlab.deployment.tasks import configure, make_contract, prepare_many, prepare_one

SOURCE = Path(__file__).resolve().parents[1]


def raw_hopper(path, obs_dim=11):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    arrays = {"observations": rng.normal(size=(20, obs_dim)).astype(np.float32),
              "actions": rng.uniform(-1, 1, (20, 3)).astype(np.float32),
              "rewards": np.zeros(20, np.float32), "terminals": np.zeros(20, bool),
              "timeouts": np.zeros(20, bool)}
    arrays["timeouts"][[9, 19]] = True
    with h5py.File(path, "w") as h5:
        for k, v in arrays.items():
            h5[k] = v
    return path


def settings(root, **updates):
    base = {"root": str(root), "profile": "d4rl", "envs": ["hopper-medium-v2"],
            "suite12": False, "device": "cpu", "code_mode": "disabled", "image": None,
            "model": "REPLACE_WITH_YOUR_MODEL", "provider": "compatible",
            "base_url": "http://127.0.0.1:8000/v1", "smoke_steps": 3}
    return {**base, **updates}


def test_default_is_d4rl_and_dry_run_has_no_side_effect(project_tmp_path, monkeypatch):
    tmp_path = project_tmp_path
    root = tmp_path / "new deployment"
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("dry-run subprocess"))
    assert main(["--root", str(root), "--dry-run", "--suite12"], source=SOURCE) == 0
    assert not root.exists()
    args = parser().parse_args([])
    assert args.profile == "d4rl" and args.code_mode == "trusted_local"
    assert len(plan(parser().parse_args(["--suite12"]), SOURCE, root)["datasets"]) == 12


def test_noninteractive_requires_consent(project_tmp_path, monkeypatch):
    tmp_path = project_tmp_path
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    root = tmp_path / "new"
    assert main(["--root", str(root)], source=SOURCE) == 1
    assert not root.exists()


@pytest.mark.parametrize("flags", [["--code-mode", "docker"],
    ["--profile", "d4rl", "--reuse-runtime", sys.executable],
    ["--base-url", "http://remote.example/v1"], ["--base-url", "https://secret:pass@remote.example/v1"],
    ["--d4rl-ref", "--upload-pack=evil"], ["--smoke-steps", "0"]])
def test_unsafe_or_inconsistent_options_rejected(project_tmp_path, flags):
    tmp_path = project_tmp_path
    with pytest.raises((ValueError, SystemExit)):
        args = parser().parse_args(["--root", str(tmp_path / "root"), *flags])
        validate_options(args, SOURCE)


def test_root_cannot_overwrite_source():
    with pytest.raises(ValueError, match="source"):
        validate_options(parser().parse_args(["--root", str(SOURCE)]), SOURCE)


def test_process_environment_excludes_model_secrets(monkeypatch):
    monkeypatch.setenv("ARLAB_API_KEY", "do-not-leak")
    monkeypatch.setenv("OPENAI_API_KEY", "do-not-leak")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "do-not-leak")
    env = public_environment()
    assert all("API_KEY" not in k for k in env)
    assert env["PIP_CONFIG_FILE"] == os.devnull


def test_dataset_offline_import_reuse_and_tamper_detection(tmp_path):
    source = raw_hopper(tmp_path / "raw.hdf5")
    target = tmp_path / "hopper-medium-v2.npz"
    first = prepare_one("hopper-medium-v2", target, raw=source, offline=True)
    second = prepare_one("hopper-medium-v2", target, raw=source, offline=True)
    assert not first["reused"] and second["reused"]
    assert first["sha256"] == second["sha256"]
    receipt = target.with_name(target.name + ".preparation.json")
    invalid = read_json(receipt)
    invalid["sha256"] = "0" * 64
    atomic_json(receipt, invalid)
    with pytest.raises(ValueError, match="changed"):
        prepare_one("hopper-medium-v2", target, raw=source, offline=True)


def test_mismatched_observation_dimension_rejected(tmp_path):
    source = raw_hopper(tmp_path / "raw.hdf5", obs_dim=17)
    with pytest.raises(ValueError, match="dimensions"):
        prepare_one("hopper-medium-v2", tmp_path / "out.npz", raw=source, offline=True)


def test_missing_data_never_falls_back_to_synthetic(tmp_path, monkeypatch):
    monkeypatch.setattr("arlab.deployment.tasks.download_raw", lambda *a: pytest.fail("offline download"))
    with pytest.raises(FileNotFoundError):
        prepare_many(tmp_path / "data", ["hopper-medium-v2"], offline=True)
    assert read_json(tmp_path / "data/datasets.json")["status"] == "failed"


def test_hash_mismatch_before_conversion(tmp_path):
    source = raw_hopper(tmp_path / "raw.hdf5")
    target = tmp_path / "out.npz"
    with pytest.raises(ValueError, match="checksum"):
        prepare_one("hopper-medium-v2", target, raw=source, expected_sha256="0" * 64)
    assert not target.exists()


def test_batch_records_partial_completion_and_retry(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_hopper(raw_dir / "hopper_medium-v2.hdf5")
    envs = ["hopper-medium-v2", "hopper-random-v2"]
    with pytest.raises(FileNotFoundError):
        prepare_many(tmp_path / "data", envs, raw_dir=raw_dir, offline=True)
    report = read_json(tmp_path / "data/datasets.json")
    assert report["status"] == "failed" and len(report["datasets"]) == 1
    raw_hopper(raw_dir / "hopper_random-v2.hdf5")
    result = prepare_many(tmp_path / "data", envs, raw_dir=raw_dir, offline=True)
    assert result["status"] == "ready" and result["datasets"][0]["reused"]


def test_config_cpu_and_no_overwrite_of_owner_edits(tmp_path):
    source = raw_hopper(tmp_path / "raw.hdf5")
    prepare_one("hopper-medium-v2", tmp_path / "data/hopper-medium-v2.npz", raw=source)
    cfg = settings(tmp_path)
    atomic_json(tmp_path / "settings.json", cfg)
    configure(tmp_path)
    path = tmp_path / "configs/research.json"
    raw = read_json(path)
    assert raw["device"] == "cpu" and not raw["code_execution"]["container_gpu"]
    raw["model"]["model"] = "owner-edited-model"
    atomic_json(path, raw)
    configure(tmp_path)
    assert read_json(path)["model"]["model"] == "owner-edited-model"
    cfg["device"] = "cuda"
    atomic_json(tmp_path / "settings.json", cfg)
    with pytest.raises(ValueError, match="did not overwrite"):
        configure(tmp_path)
    assert read_json(path)["device"] == "cpu"
    assert (path.parent / "research.proposed.json").exists()


def test_failed_stage_persists_failure(tmp_path):
    args = parser().parse_args(["--profile", "smoke", "--reuse-runtime", sys.executable])
    deploy = Deployer(SOURCE, tmp_path, args)
    with pytest.raises(RuntimeError):
        deploy.stage("injected-error", lambda: (_ for _ in ()).throw(RuntimeError("test")))
    state = read_json(tmp_path / "deployment.json")
    assert state["status"] == "failed" and state["steps"][-1]["status"] == "failed"


def test_skip_smoke_never_claims_ready(tmp_path, monkeypatch):
    args = parser().parse_args(["--profile", "smoke", "--reuse-runtime", sys.executable, "--skip-smoke"])
    deploy = Deployer(SOURCE, tmp_path, args)
    for name in ("preflight", "runtime", "configure", "checks"):
        monkeypatch.setattr(deploy, name, lambda: None)
    deploy.run()
    assert read_json(tmp_path / "deployment.json")["status"] == "prepared_unverified"


def test_local_backend_keeps_headless_mode_but_not_keys(tmp_path, monkeypatch):
    from arlab.core.jobs import LocalBackend
    monkeypatch.setenv("MUJOCO_PY_FORCE_CPU", "1")
    monkeypatch.setenv("ARLAB_API_KEY", "secret-value")
    command = [sys.executable, "-c", "import os; assert os.environ.get('MUJOCO_PY_FORCE_CPU') == '1'; assert 'ARLAB_API_KEY' not in os.environ"]
    assert LocalBackend().run(command, tmp_path, tmp_path / "log", 20).exit_code == 0


@pytest.mark.integration
def test_actual_no_network_deployment_and_safe_launcher(project_tmp_path):
    tmp_path = project_tmp_path
    # Explicitly reuses installed dependencies; does not claim a fresh pip/MuJoCo install.
    root = tmp_path / "deployment with spaces"
    result = subprocess.run(["bash", str(SOURCE / "deploy.sh"), "--profile", "smoke", "--root", str(root),
                             "--reuse-runtime", sys.executable, "--yes", "--smoke-steps", "2"],
                            text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    state = read_json(root / "deployment.json")
    assert state["status"] == "ready" and state["llm_calls"] == 0
    assert state["installation"] == "skipped_owner_supplied_runtime"
    assert read_json(root / "smoke-latest.json")["status"] == "passed"
    cli = subprocess.run([str(root / "bin/arlab"), "doctor"], text=True, capture_output=True, timeout=30)
    assert cli.returncode == 0, cli.stderr
    launch = subprocess.run([str(root / "start-research.sh")], text=True, capture_output=True, timeout=30)
    assert launch.returncode == 1 and "confirm-budget" in launch.stderr
    assert not (root / "runs/research").exists()
    # Repeat the whole deployment: no environment downgrade or config overwrite.
    config = read_json(root / "configs/research.json")
    config["model"]["model"] = "owner-customized"
    atomic_json(root / "configs/research.json", config)
    again = subprocess.run(["bash", str(SOURCE / "deploy.sh"), "--profile", "smoke", "--root", str(root),
                            "--reuse-runtime", sys.executable, "--yes", "--skip-smoke"],
                           text=True, capture_output=True, timeout=60)
    assert again.returncode == 0, again.stdout + again.stderr
    assert read_json(root / "configs/research.json")["model"]["model"] == "owner-customized"
    assert read_json(root / "deployment.json")["status"] == "prepared_unverified"


def test_docker_policy_client_keeps_owner_context_without_exposing_model_keys(monkeypatch):
    from arlab.algorithms.policy import policy_process_environment
    monkeypatch.setenv("DOCKER_HOST", "unix:///run/user/1000/docker.sock")
    monkeypatch.setenv("DOCKER_CONFIG", "/owner/.docker")
    monkeypatch.setenv("ARLAB_API_KEY", "never-pass-me")
    container_client = policy_process_environment("/temporary", "docker")
    local_candidate = policy_process_environment("/temporary", "trusted_local")
    assert container_client["DOCKER_HOST"].endswith("docker.sock")
    assert container_client["DOCKER_CONFIG"] == "/owner/.docker"
    assert "DOCKER_HOST" not in local_candidate
    assert "ARLAB_API_KEY" not in container_client and "ARLAB_API_KEY" not in local_candidate


def test_local_timeout_allows_finally_cleanup(tmp_path):
    from arlab.core.jobs import LocalBackend
    script = """import signal,time
from pathlib import Path
def interrupt(sig, frame):
    raise KeyboardInterrupt
signal.signal(signal.SIGTERM, interrupt)
try:
    time.sleep(60)
finally:
    Path('cleanup-ran').write_text('yes')
"""
    result = LocalBackend().run([sys.executable, "-S", "-c", script], tmp_path, tmp_path / "worker.log", 1.0)
    assert result.timed_out and result.exit_code != 0
    assert (tmp_path / "cleanup-ran").read_text() == "yes"


def test_deployment_direct_pins_cover_separate_legacy_and_torch():
    from arlab.deployment.manager import direct_pins
    smoke = direct_pins(SOURCE, "smoke")
    d4rl = direct_pins(SOURCE, "d4rl")
    assert smoke["torch"] == "2.5.1" and "mujoco-py" not in smoke
    assert d4rl["mujoco-py"] == "2.1.2.14"
    assert d4rl["Cython"] == "0.29.37"
    assert all(d4rl[k] == v for k, v in smoke.items())
