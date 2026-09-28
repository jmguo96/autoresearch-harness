"""Project-local storage for the installer and its child processes.

This is dependency/configuration isolation, NOT a security sandbox. The same
Unix user can still access other projects. Never describe a venv as filesystem
or network confinement for generated Python.
"""
from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import Iterator


LOCAL_CODE_NOTICE = (
    "本版本使用项目内 uv 虚拟环境执行 Python，不使用 Docker 或 sudo。"
    "虚拟环境只隔离依赖，不隔离文件或网络权限；生成代码仍有当前用户权限。"
    "请审阅候选代码，不要以 root 运行，也不要在含敏感资料的账号中运行未审阅代码。"
)


def contained_path(path: Path, parent: Path, *, allow_parent: bool = False) -> Path:
    """Reject an external destination or an existing symlink on its path.

    Read-only input files may live elsewhere; this helper is for managed output
    directories. It is a preflight check, not protection against hostile races.
    """
    parent = parent.absolute().resolve()
    path = path.expanduser().absolute()
    if not path.is_relative_to(parent) or (path == parent and not allow_parent):
        raise ValueError("Managed output must be inside the project source directory")
    cursor = parent
    for part in path.relative_to(parent).parts:
        if part in ("..", "."):
            raise ValueError("Managed output cannot contain parent traversal")
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"Managed output cannot pass through a symlink: {cursor}")
    if not path.resolve().is_relative_to(parent):
        raise ValueError("Managed output escapes the project source directory")
    return path


def project_environment(root: Path) -> dict[str, str]:
    """Pure mapping; does not write anything (also usable by --dry-run)."""
    root = root.absolute()
    dirs = {
        "ARLAB_LOCAL_ROOT": root,
        "CPATH": root / "native/include",
        "HOME": root / "home",
        "XDG_CACHE_HOME": root / "cache",
        "XDG_CONFIG_HOME": root / "config",
        "XDG_CONFIG_DIRS": root / "config",
        "XDG_DATA_HOME": root / "share",
        "XDG_STATE_HOME": root / "state",
        "XDG_RUNTIME_DIR": root / "tmp/runtime",
        "TMPDIR": root / "tmp", "TMP": root / "tmp", "TEMP": root / "tmp",
        "UV_CACHE_DIR": root / "cache/uv",
        "UV_PYTHON_INSTALL_DIR": root / "python",
        "UV_PYTHON_BIN_DIR": root / "python-bin",
        "UV_PYTHON_CACHE_DIR": root / "cache/uv-python",
        "UV_TOOL_DIR": root / "tools/environments",
        "UV_TOOL_BIN_DIR": root / "tools/bin",
        "UV_CREDENTIALS_DIR": root / "config/uv-credentials",
        "PIP_CACHE_DIR": root / "cache/pip",
        "PYTHONUSERBASE": root / "python-user",
        "MPLCONFIGDIR": root / "cache/matplotlib",
        "TORCH_HOME": root / "cache/torch",
        "TORCH_EXTENSIONS_DIR": root / "cache/torch-extensions",
        "TORCHINDUCTOR_CACHE_DIR": root / "cache/torchinductor",
        "TRITON_CACHE_DIR": root / "cache/triton",
        "CUDA_CACHE_PATH": root / "cache/cuda",
        "NUMBA_CACHE_DIR": root / "cache/numba",
        "HF_HOME": root / "cache/huggingface",
        "HF_HUB_CACHE": root / "cache/huggingface/hub",
        "HF_DATASETS_CACHE": root / "cache/huggingface/datasets",
        "WANDB_DIR": root / "cache/wandb",
        "WANDB_CACHE_DIR": root / "cache/wandb/cache",
        "WANDB_CONFIG_DIR": root / "config/wandb",
        "D4RL_DATASET_DIR": root / "data/raw",
    }
    result = {key: str(value) for key, value in dirs.items()}
    # Auto-provisioned .deb libraries are a curated overlay, NOT a whole sysroot
    # on LD_LIBRARY_PATH. The host libc / ELF loader remain in use.
    result["CPATH"] = os.pathsep.join(str(root / x) for x in ("native/include", "native/auto/include"))
    native_libs = os.pathsep.join(str(root / x) for x in ("native/lib", "native/lib64", "native/auto/lib"))
    result["LIBRARY_PATH"] = native_libs
    result["LD_LIBRARY_PATH"] = native_libs
    result.update(
        UV_NO_CONFIG="1", UV_NO_MODIFY_PATH="1", UV_NO_ENV_FILE="1",
        UV_LINK_MODE="copy", UV_KEYRING_PROVIDER="disabled",
        PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
        PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK="1",
        PIP_REQUIRE_VIRTUALENV="1", GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_COUNT="0", GIT_TERMINAL_PROMPT="0",
        WANDB_MODE="disabled",
    )
    return result


def prepare_local_directories(root: Path) -> None:
    """Create only project-managed paths, refusing pre-existing link redirects."""
    root = root.absolute()
    paths = set()
    for key, value in project_environment(root).items():
        paths.update(value.split(os.pathsep) if key in ("CPATH", "LIBRARY_PATH", "LD_LIBRARY_PATH") else [value])
    for value in paths:
        path = Path(value)
        if not path.is_absolute() or not path.is_relative_to(root):
            continue
        contained_path(path, root, allow_parent=True)
        if path.is_relative_to(root / "native/auto"):
            continue  # Published atomically only after extraction and validation.
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    for relative in ("downloads", "tools/uv/bin", "native/include", "native/lib", "native/bin"):
        path = contained_path(root / relative, root)
        path.mkdir(parents=True, exist_ok=True, mode=0o700)


def clean_path(root: Path, inherited: str | None = None) -> str:
    """Do not use tools from an unrelated activated venv/Conda by accident."""
    prefixes = [os.environ.get(k) for k in ("VIRTUAL_ENV", "CONDA_PREFIX")]
    excluded = {str(Path(p) / "bin") for p in prefixes if p}
    base = inherited if inherited is not None else os.environ.get("PATH", os.defpath)
    items = [str(root / "venv/bin"), str(root / "tools/uv/bin"), str(root / "native/bin")]
    items += [str(Path(p).absolute()) for p in base.split(os.pathsep) if p and p not in excluded]
    return os.pathsep.join(dict.fromkeys(items))


@contextlib.contextmanager
def local_process(root: Path, source: Path) -> Iterator[None]:
    """Scope environment changes to this process tree; never edit shell profiles."""
    root = contained_path(root, source)
    old = dict(os.environ)
    previous_temp = tempfile.tempdir
    try:
        path = clean_path(root)
        # Do not inherit installation targets, Docker settings or project hooks.
        for key in list(os.environ):
            if key.startswith(("UV_", "PIP_", "CONDA_", "DOCKER_", "GIT_CONFIG_")) or key in (
                "VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP",
                "PYTHONPYCACHEPREFIX", "BASH_ENV", "ENV", "LD_PRELOAD",
            ):
                os.environ.pop(key, None)
        os.environ.update(project_environment(root))
        os.environ["ARLAB_PROJECT_ROOT"] = str(source.resolve())
        os.environ["PATH"] = path
        prepare_local_directories(root)
        tempfile.tempdir = None  # Respect the newly scoped TMPDIR even after prior use.
        yield
    finally:
        os.environ.clear()
        os.environ.update(old)
        tempfile.tempdir = previous_temp


def inherit_local_environment(env: dict[str, str]) -> dict[str, str]:
    """Restore local paths after a credential-scrubbing worker allowlist.

    No API keys are copied here. Legacy callers outside a managed project retain
    their existing behavior; the project launchers always set ARLAB_LOCAL_ROOT.
    """
    root = os.environ.get("ARLAB_LOCAL_ROOT")
    if root:
        env.update(project_environment(Path(root)))
        if "ARLAB_PROJECT_ROOT" in os.environ:
            env["ARLAB_PROJECT_ROOT"] = os.environ["ARLAB_PROJECT_ROOT"]
        for key in ("LIBRARY_PATH", "LD_LIBRARY_PATH"):
            if key in os.environ:
                env[key] = os.environ[key]
        for key in tuple(env):
            if key.startswith("DOCKER_"):
                env.pop(key, None)
    return env
