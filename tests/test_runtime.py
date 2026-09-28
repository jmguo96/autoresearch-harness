from __future__ import annotations

import json
import sys

import httpx
import pytest

from arlab.core.io import canonical, read_json, within, write_json
from arlab.core.jobs import LocalBackend
from arlab.core.models import HTTPModel
from arlab.core.schema import Decision, ModelConfig, PlanAction
from arlab.core.stats import interval, iqm, paired_comparison
from arlab.core.store import BudgetExceeded, Store, campaign_lock


def test_atomic_json_and_finite_values(tmp_path):
    path = tmp_path / "nested/test.json"
    write_json(path, {"a": 1})
    assert read_json(path) == {"a": 1}
    with pytest.raises(ValueError):
        write_json(path, {"a": float("nan")})
    assert read_json(path) == {"a": 1}


@pytest.mark.parametrize("name", ["../escape", "/etc/passwd", "a/../../x"])
def test_path_traversal_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        within(tmp_path, name)


def test_symlink_escape_rejected(tmp_path):
    (tmp_path / "escape").symlink_to("/etc", target_is_directory=True)
    with pytest.raises(ValueError):
        within(tmp_path, "escape/passwd")


def test_transactional_budget_and_idempotence(tmp_path):
    store = Store(tmp_path / "s.db")
    try:
        store.reserve("a", {"x": 2}, {"x": 3})
        store.reserve("a", {"x": 2}, {"x": 3})
        assert store.totals() == {"x": 2}
        with pytest.raises(BudgetExceeded):
            store.reserve("b", {"y": 1, "x": 2}, {"x": 3, "y": 10})
        assert store.totals() == {"x": 2}  # no partial reservation
        with pytest.raises(ValueError):
            store.reserve("a", {"x": 1}, {"x": 3})
    finally:
        store.close()


def test_memory_requires_evidence_and_supports_revision(tmp_path):
    store = Store(tmp_path / "s.db")
    try:
        with pytest.raises(ValueError):
            store.remember("fabricated finding", "hypothesis", ["t-does-not-exist"])
        eid = store.event("observation", {"failed": True})
        old = store.remember("IQL failure needs checking", "procedure", [eid])
        store.remember("IQL failure was numerical; improvement remains unproven", "procedure", [eid], supersedes=old)
        retrieved = store.recall("IQL")
        assert len(retrieved) == 1 and not retrieved[0]["verified_claim"]
    finally:
        store.close()


def test_campaign_lock_rejects_second_owner(tmp_path):
    with campaign_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another"):
            with campaign_lock(tmp_path):
                pass


def test_pending_actions_survive_reopen(tmp_path):
    path = tmp_path / "s.db"
    store = Store(path)
    aid = store.begin_action({"test": 1})
    store.close()
    store = Store(path)
    try:
        assert store.pending_action() == (aid, {"test": 1})
        store.finish_action(aid, {"ok": True})
        assert store.pending_action() is None
    finally:
        store.close()


def test_plan_dependencies_and_cycles():
    with pytest.raises(ValueError):
        PlanAction(kind="plan", reason="test", items=[{"id": "a", "objective": "x", "depends_on": ["a"]}])
    with pytest.raises(ValueError):
        PlanAction(kind="plan", reason="test", items=[{"id": "a", "objective": "x", "depends_on": ["b"]}])


def test_no_arbitrary_shell_tool():
    with pytest.raises(ValueError):
        Decision.model_validate({"summary": "test", "action": {"kind": "shell", "command": "rm -rf /"}})


def test_local_worker_scrubs_model_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("ARLAB_API_KEY", "TOP_SECRET")
    log = tmp_path / "log"
    result = LocalBackend().run([sys.executable, "-c", "import os; print(os.environ.get('ARLAB_API_KEY', 'clean'))"], tmp_path, log, 5)
    assert result.exit_code == 0 and log.read_text().strip() == "clean"


def test_worker_timeout_is_enforced(tmp_path):
    result = LocalBackend().run([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path, tmp_path / "log", 0.15)
    assert result.timed_out and result.exit_code != 0


@pytest.mark.parametrize("provider", ["compatible", "anthropic"])
def test_model_adapters_validate_actual_protocols(provider, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "secret")
    payload = {"summary": "inspect", "action": {"kind": "inspect", "target": "skill", "id": "iql"}}
    calls = []
    def handler(request):
        calls.append(request)
        if provider == "compatible":
            assert request.headers["authorization"] == "Bearer secret"
            body = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(payload)}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 20}}
        else:
            assert request.headers["x-api-key"] == "secret"
            body = {"content": [{"type": "text", "text": json.dumps(payload)}],
                    "usage": {"input_tokens": 10, "output_tokens": 20}}
        return httpx.Response(200, json=body)
    model = HTTPModel(ModelConfig(provider=provider, model="test", base_url="https://example.test/v1", api_key_env="TEST_KEY"),
                      transport=httpx.MockTransport(handler))
    reply = model.complete("system", {"goal": "test"}, Decision)
    assert reply.payload == Decision.model_validate(payload).model_dump(mode="json") and reply.output_tokens == 20 and len(calls) == 1


def test_auth_failure_is_not_retried():
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(401, json={"error": "invalid"})
    model = HTTPModel(ModelConfig(provider="compatible", base_url="https://example.test/v1"), transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError):
        model.complete("s", {}, Decision)
    assert len(calls) == 1


def test_remote_plaintext_model_endpoint_is_rejected():
    model = HTTPModel(ModelConfig(provider="compatible", base_url="http://untrusted.example/v1"))
    with pytest.raises(ValueError, match="HTTPS"):
        model.complete("s", {}, Decision)


def test_single_seed_has_no_confidence_interval():
    assert interval([1])["lower"] is None
    assert iqm([1, 2, 3, 4]) == 2.5
    assert iqm([3]) == 3


def test_comparison_does_not_mix_horizons_or_incomplete_panels():
    baseline = [{"case": c, "train_seed": s, "steps": 10, "metric": "return", "score": 1.0}
                for c in ["a", "b"] for s in [0, 1]]
    candidate = [{**t, "score": 2.0} for t in baseline]
    assert paired_comparison(baseline, candidate)["status"] == "positive_evidence"
    candidate[0]["steps"] = 20
    result = paired_comparison(baseline, candidate)
    assert not result["complete"] and result["status"] == "insufficient_evidence"


def test_nested_model_fingerprints_are_canonical():
    from arlab.core.io import digest
    from arlab.core.config import AgentConfig
    model = AgentConfig()
    assert digest({"systems": [model]}) == digest({"systems": [model.model_dump(mode="json")]})


def test_worker_can_import_package_after_changing_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "src")
    result = LocalBackend().run([sys.executable, "-c", "import arlab; print(arlab.__version__)"], tmp_path, tmp_path / "log", 5)
    assert result.exit_code == 0


@pytest.mark.parametrize("url", ["http://localhost.evil.example/v1", "http://127.0.0.1.evil.example/v1",
                                  "https://name:password@example.com/v1"])
def test_model_endpoint_disallows_lookalike_localhost_and_url_credentials(url):
    from arlab.core.models import HTTPModel
    from arlab.core.schema import ModelConfig, Decision
    with pytest.raises(ValueError):
        HTTPModel(ModelConfig(provider="compatible", base_url=url)).complete("rules", {}, Decision)
