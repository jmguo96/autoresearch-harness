from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(autouse=True, scope="session")
def bounded_torch_threads():
    import torch
    torch.set_num_threads(1)


@pytest.fixture
def dataset(tmp_path):
    from arlab.tasks.iql.environments import make_fixture
    path = tmp_path / "data.npz"
    sha = make_fixture(path, episodes=8, horizon=8)
    return path, sha


@pytest.fixture
def project_tmp_path():
    """Managed installation roots must be children of the actual source tree."""
    import tempfile
    base = Path(__file__).resolve().parents[1] / ".arlab" / "test-workspaces"
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="case-", dir=base) as temporary:
        yield Path(temporary)
