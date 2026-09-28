"""Gateway tests use synthetic HTTP responses and fake keys, not a live lab account."""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError

import httpx
import pytest

from arlab.classroom import kimi, session
from arlab.classroom.bootstrap import main as bootstrap_main
from arlab.core.endpoints import AUTO_KIMI, LAB_NEW_API_BASE_URL, validate_model_endpoint
from arlab.core.models import HTTPModel, ModelBusy
from arlab.core.schema import Decision, ModelConfig
from arlab.deployment.common import public_environment
from arlab.core.jobs import LocalBackend
from arlab.algorithms.policy import policy_process_environment

FAKE_KEY = "sk-test-fixture-NOT-a-real-key"


def config(**kwargs):
    raw = kimi.default_model_settings()
    raw.update(kwargs)
    return ModelConfig.model_validate(raw)


class Response(io.BytesIO):
    status = 200


class StubOpener:
    """In-memory protocol fixture; records requests without contacting the network."""
    def __init__(self, data=None, *, raw=None, error=None):
        self.raw = raw if raw is not None else json.dumps(data).encode()
        self.error = error
        self.requests = []

    def open(self, request, timeout):
        self.requests.append((request, timeout))
        if self.error:
            raise self.error
        return Response(self.raw)


def model_list(*ids):
    return {"object": "list", "data": [{"id": value} for value in ids]}


def test_default_settings_need_only_key_and_preserve_research():
    c = config()
    assert c.model == AUTO_KIMI and c.provider == "compatible"
    assert c.base_url == LAB_NEW_API_BASE_URL and c.api_key_env == "NEW_API_KEY"
    assert c.allow_lab_http and c.json_mode
    assert "api_key" not in c.model_dump()


@pytest.mark.parametrize("ids, expected", [
    (["gpt-x", "kimi-k2.5", "kimi"], "kimi"),
    (["k3", "glm-5.3", "glm-5.3-flash"], "k3"),
    (["kimi-k2.5", "kimi-k2.6"], "kimi-k2.6"),
    (["kimi-k2.9", "kimi-k2.10"], "kimi-k2.10"),
    (["moonshotai/Kimi-K2-Instruct", "kimi-audio", "kimi-k2-base"], "moonshotai/Kimi-K2-Instruct"),
    (["kimi-latest", "kimi-k3"], "kimi-latest"),
])
def test_selection_uses_only_advertised_ids(ids, expected):
    assert kimi.choose_kimi(ids) == expected
    assert kimi.choose_kimi(list(reversed(ids))) == expected


@pytest.mark.parametrize("ids", [[], ["gpt-x", "claude-x"], ["moonshot-v1-128k"], ["k2", "k4", "k3-audio"],
    ["kimi-audio", "kimi-k2-base"], ["not-kimi-k3", "kimi\nsecret", None, 7]])
def test_no_kimi_never_falls_back(ids):
    with pytest.raises(ValueError, match="Kimi"):
        kimi.choose_kimi(ids)


def test_pinned_model_is_never_upgraded_or_guessed():
    assert kimi.choose_kimi(["k3", "glm-5.3"], "k3") == "k3"
    assert kimi.choose_kimi(["kimi-k2", "kimi-k3"], "kimi-k2") == "kimi-k2"
    with pytest.raises(ValueError, match="固定"):
        kimi.choose_kimi(["kimi-k3"], "kimi-k2")
    with pytest.raises(ValueError):
        kimi.choose_kimi(["gpt-x"], "gpt-x")


@pytest.mark.parametrize("url", [
    "http://43.143.241.66:18320/v1", "http://43.143.241.67:18319/v1",
    "http://43.143.241.66:18319/admin", "http://evil.example/v1",
    "http://user:secret@43.143.241.66:18319/v1", LAB_NEW_API_BASE_URL + "?key=secret",
    LAB_NEW_API_BASE_URL + "#fragment", "http://43.143.241.66:18319/v1\n",
])
def test_http_grant_is_origin_and_path_specific(url):
    with pytest.raises(ValueError, match="HTTPS"):
        validate_model_endpoint(url, allow_lab_http=True)


def test_plaintext_gateway_requires_explicit_grant():
    with pytest.raises(ValueError):
        validate_model_endpoint(LAB_NEW_API_BASE_URL)
    assert not validate_model_endpoint(LAB_NEW_API_BASE_URL, allow_lab_http=True)
    assert validate_model_endpoint("http://127.0.0.1:8000/v1")
    assert not validate_model_endpoint("https://teacher.example/v1")


def test_discovery_request_protocol():
    opener = StubOpener(model_list("kimi-k2.5"))
    assert kimi.discover_kimi(FAKE_KEY, opener=opener) == "kimi-k2.5"
    req, timeout = opener.requests[0]
    assert req.full_url == LAB_NEW_API_BASE_URL + "/models"
    assert req.get_method() == "GET" and req.get_header("Authorization") == "Bearer " + FAKE_KEY
    assert timeout == 20 and req.data is None


@pytest.mark.parametrize("status", [301, 302, 401, 403, 404, 429, 503])
def test_http_errors_redact_bodies_and_do_not_retry(status):
    err = HTTPError(LAB_NEW_API_BASE_URL, status, "secret " + FAKE_KEY, {}, io.BytesIO(FAKE_KEY.encode()))
    opener = StubOpener(error=err)
    with pytest.raises(ValueError) as caught:
        kimi.discover_kimi(FAKE_KEY, opener=opener)
    assert FAKE_KEY not in str(caught.value)
    assert len(opener.requests) == 1


@pytest.mark.parametrize("raw", [b"<html>error</html>", b"[]", b'{"data":null}', b"\xff", b"a" * 1_000_001])
def test_invalid_or_oversized_models_fail_closed(raw):
    with pytest.raises(ValueError):
        kimi.discover_kimi(FAKE_KEY, opener=StubOpener(raw=raw))


def test_network_error_does_not_echo_credentials():
    with pytest.raises(ValueError) as caught:
        kimi.discover_kimi(FAKE_KEY, opener=StubOpener(error=URLError(FAKE_KEY)))
    assert FAKE_KEY not in str(caught.value)


def test_interactive_settings_do_not_request_host_or_model(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.stdin", type("TTY", (), {"isatty": lambda self: True})())
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("Must not ask for host/model"))
    args = argparse.Namespace(action="run", provider=None, model=None, base_url=None)
    c = session.model_settings(tmp_path, args, ModelConfig())
    assert c == config()
    assert (tmp_path / "model.json").stat().st_mode & 0o777 == 0o600
    assert FAKE_KEY not in (tmp_path / "model.json").read_text()


def test_only_hidden_key_prompt(tmp_path, monkeypatch):
    monkeypatch.delenv("NEW_API_KEY", raising=False)
    monkeypatch.delenv("ARLAB_API_KEY", raising=False)
    monkeypatch.setattr("sys.stdin", type("TTY", (), {"isatty": lambda self: True})())
    prompts = []
    monkeypatch.setattr(session.getpass, "getpass", lambda msg: prompts.append(msg) or FAKE_KEY)
    try:
        assert session.require_key(config(), False)
        assert os.environ["NEW_API_KEY"] == FAKE_KEY and len(prompts) == 1
    finally:
        os.environ.pop("NEW_API_KEY", None)


def test_legacy_key_alias_is_temporary(monkeypatch):
    monkeypatch.delenv("NEW_API_KEY", raising=False)
    monkeypatch.setenv("ARLAB_API_KEY", FAKE_KEY)
    try:
        assert session.require_key(config(), False)
        assert os.environ["NEW_API_KEY"] == FAKE_KEY
    finally:
        os.environ.pop("NEW_API_KEY", None)


@pytest.mark.parametrize("key", ["", "secret\nheader", "带中文", "has space", "x" * 4097])
def test_bad_key_rejected_without_echo(key):
    with pytest.raises(ValueError):
        kimi.validate_key(key)


def test_real_http_discovery_bypasses_proxy_and_refuses_redirect(monkeypatch):
    """Real loopback HTTP, synthetic model catalogue, no external requests."""
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append(self.path)
            assert self.headers.get("Authorization") == "Bearer " + FAKE_KEY
            if self.path == "/redirect/models":
                self.send_response(302)
                self.send_header("Location", "/stolen")
                self.end_headers()
                return
            raw = json.dumps(model_list("kimi-fixture", "kimi-k2.5")).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    try:
        assert kimi.discover_kimi(FAKE_KEY, base_url=base + "/v1") == "kimi-k2.5"
        with pytest.raises(ValueError, match="重定向"):
            kimi.discover_kimi(FAKE_KEY, base_url=base + "/redirect")
        assert calls == ["/v1/models", "/redirect/models"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_chat_completions_protocol_ignores_separate_reasoning(monkeypatch):
    monkeypatch.setenv("NEW_API_KEY", FAKE_KEY)
    decision = {"summary": "fixture", "action": {"kind": "inspect", "target": "skill", "id": "iql"}}
    seen = []
    def handler(request):
        seen.append(request)
        body = json.loads(request.content)
        assert request.url == LAB_NEW_API_BASE_URL + "/chat/completions"
        assert request.headers["Authorization"] == "Bearer " + FAKE_KEY
        assert body["model"] == "kimi-k2.5" and body["stream"] is False
        assert body["response_format"] == {"type": "json_object"}
        assert "thinking" not in body and "temperature" not in body
        return httpx.Response(200, json={"choices": [{"message": {
            "content": json.dumps(decision), "reasoning_content": "PRIVATE_FIXTURE_NOT_RETURNED"},
            "finish_reason": "stop"}], "usage": {"prompt_tokens": 12, "completion_tokens": 25}})
    reply = HTTPModel(config(model="kimi-k2.5"), transport=httpx.MockTransport(handler)).complete("rules", {}, Decision)
    assert reply.payload == Decision.model_validate(decision).model_dump(mode="json") and reply.input_tokens == 12
    assert "PRIVATE_FIXTURE" not in repr(reply)
    assert len(seen) == 1


def test_responses_protocol_uses_list_input_stream_and_parses_sse(monkeypatch):
    monkeypatch.setenv("NEW_API_KEY", FAKE_KEY)
    seen = []
    events = [
        {"type": "response.output_text.delta", "delta": '{"summary":"fixture",'},
        {"type": "response.output_text.delta", "delta": '"action":{"kind":"inspect","target":"skill","id":"iql"}}'},
        {"type": "response.completed", "response": {
            "usage": {"input_tokens": 12, "output_tokens": 25}}},
    ]
    content = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events) + b"data: [DONE]\n\n"
    def handler(request):
        seen.append(request)
        body = json.loads(request.content)
        assert request.url == LAB_NEW_API_BASE_URL + "/responses"
        assert body["model"] == "gpt-5.6-sol" and body["stream"] is True
        assert isinstance(body["input"], list) and body["input"][0]["role"] == "developer"
        return httpx.Response(200, content=content, headers={"content-type": "text/event-stream"})
    reply = HTTPModel(config(provider="responses", model="gpt-5.6-sol"),
                      transport=httpx.MockTransport(handler)).complete("rules", {}, Decision)
    assert reply.payload["action"]["kind"] == "inspect"
    assert reply.input_tokens == 12 and reply.output_tokens == 25 and len(seen) == 1


def test_lab_transient_400_retries_without_optional_json_mode(monkeypatch):
    monkeypatch.setenv("NEW_API_KEY", FAKE_KEY)
    monkeypatch.setattr("arlab.core.models.time.sleep", lambda _: None)
    decision = {"summary": "fixture", "action": {"kind": "inspect", "target": "skill", "id": "iql"}}
    calls = []
    def handler(request):
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(400, json={"error": {"message": FAKE_KEY}})
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(decision)},
            "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 20}})
    reply = HTTPModel(config(model="k3"), transport=httpx.MockTransport(handler)).complete("rules", {}, Decision)
    assert reply.attempts == 2 and len(calls) == 2
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in calls[1]


def test_non_lab_bad_request_is_not_retried_or_disclosed(monkeypatch):
    monkeypatch.setenv("TEST_KEY", FAKE_KEY)
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(400, json={"error": {"message": FAKE_KEY}})
    model = HTTPModel(ModelConfig(provider="compatible", model="test", base_url="https://example.test/v1",
        api_key_env="TEST_KEY"), transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="HTTP 400") as caught:
        model.complete("rules", {}, Decision)
    assert len(calls) == 1 and FAKE_KEY not in str(caught.value)


def test_lab_concurrency_403_waits_and_recovers_without_disclosure(monkeypatch):
    monkeypatch.setenv("NEW_API_KEY", FAKE_KEY)
    delays = []
    monkeypatch.setattr("arlab.core.models.time.sleep", delays.append)
    decision = {"summary": "fixture", "action": {"kind": "inspect", "target": "skill", "id": "iql"}}
    calls = []
    def handler(request):
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(403, json={"error": {"type": "access_terminated_error",
                "message": "You've reached your concurrent request limit. " + FAKE_KEY}})
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(decision)},
            "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 20}})
    reply = HTTPModel(config(model="k3"), transport=httpx.MockTransport(handler)).complete("rules", {}, Decision)
    assert reply.attempts == 3 and len(calls) == 3 and delays == [60, 120]
    assert FAKE_KEY not in repr(reply)


def test_lab_concurrency_exhaustion_is_recoverable_busy_state(monkeypatch):
    monkeypatch.setenv("NEW_API_KEY", FAKE_KEY)
    monkeypatch.setattr("arlab.core.models.time.sleep", lambda _: None)
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(403, json={"error": {"type": "access_terminated_error",
            "message": "You've reached your concurrent request limit. " + FAKE_KEY}})
    with pytest.raises(ModelBusy) as caught:
        HTTPModel(config(model="k3"), transport=httpx.MockTransport(handler)).complete("rules", {}, Decision)
    assert len(calls) == 3 and FAKE_KEY not in str(caught.value)


def test_other_lab_403_body_is_also_treated_as_recoverable_capacity(monkeypatch):
    monkeypatch.setenv("NEW_API_KEY", FAKE_KEY)
    monkeypatch.setattr("arlab.core.models.time.sleep", lambda _: None)
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(403, json={"error": {"type": "permission_error", "message": FAKE_KEY}})
    with pytest.raises(ModelBusy) as caught:
        HTTPModel(config(model="k3"), transport=httpx.MockTransport(handler)).complete("rules", {}, Decision)
    assert len(calls) == 3 and FAKE_KEY not in str(caught.value)


def test_lab_empty_403_is_recoverable_capacity(monkeypatch):
    monkeypatch.setenv("NEW_API_KEY", FAKE_KEY)
    delays = []
    monkeypatch.setattr("arlab.core.models.time.sleep", delays.append)
    model = HTTPModel(config(model="k3"), transport=httpx.MockTransport(
        lambda request: httpx.Response(403, content=b"")))
    with pytest.raises(ModelBusy):
        model.complete("rules", {}, Decision)
    assert delays == [60, 120]


def test_unresolved_selector_never_sent_to_inference(monkeypatch):
    monkeypatch.setenv("NEW_API_KEY", FAKE_KEY)
    with pytest.raises(ValueError, match="Resolve"):
        HTTPModel(config(), transport=httpx.MockTransport(lambda r: pytest.fail("No request"))).complete("rules", {}, Decision)


def test_missing_channel_is_not_retried(monkeypatch):
    monkeypatch.setenv("NEW_API_KEY", FAKE_KEY)
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(503, json={"error": {"code": "model_not_found", "message": FAKE_KEY}})
    with pytest.raises(RuntimeError) as caught:
        HTTPModel(config(model="kimi-k2.5"), transport=httpx.MockTransport(handler)).complete("rules", {}, Decision)
    assert len(calls) == 1 and FAKE_KEY not in str(caught.value)


def test_key_is_not_in_installer_or_policy_environment(monkeypatch):
    monkeypatch.setenv("NEW_API_KEY", FAKE_KEY)
    monkeypatch.setenv("ARLAB_API_KEY", FAKE_KEY)
    for env in (public_environment(), policy_process_environment("/tmp", "docker"), policy_process_environment("/tmp", "trusted_local")):
        assert "NEW_API_KEY" not in env and "ARLAB_API_KEY" not in env


def test_real_worker_environment_does_not_inherit_lab_key(tmp_path, monkeypatch):
    monkeypatch.setenv("NEW_API_KEY", FAKE_KEY)
    log = tmp_path / "worker.log"
    result = LocalBackend().run([sys.executable, "-c", "import os; assert 'NEW_API_KEY' not in os.environ; print('clean')"], tmp_path, log, 10)
    assert result.exit_code == 0 and "clean" in log.read_text() and FAKE_KEY not in log.read_text()


def test_preinstall_check_api_dry_run_has_no_writes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(kimi, "discover_kimi", lambda *a, **k: pytest.fail("No network"))
    assert bootstrap_main(["check-api", "--dry-run"], source=tmp_path) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["model_selector"] == AUTO_KIMI and not list(tmp_path.iterdir())


def test_check_api_uses_pinned_responses_model(tmp_path, monkeypatch, capsys):
    root = tmp_path / ".arlab" / "d4rl9"
    root.mkdir(parents=True)
    (root / "model.json").write_text(json.dumps(config(
        provider="responses", model="gpt-5.6-sol").model_dump(mode="json")))
    monkeypatch.setenv("NEW_API_KEY", FAKE_KEY)
    opener = StubOpener(model_list("k3", "gpt-5.6-sol"))
    monkeypatch.setattr(kimi, "build_opener", lambda *args: opener)
    assert bootstrap_main(["check-api"], source=tmp_path) == 0
    output = capsys.readouterr().out
    assert "gpt-5.6-sol（responses）" in output
    assert FAKE_KEY not in output
    assert len(opener.requests) == 1


def test_check_api_requires_no_installed_packages():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-S", "-B", str(root / "scripts/classroom.py"), "check-api", "--dry-run"],
                             capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["network_request_now"] is False


def session_contract(tmp_path):
    """Contract fixture, not D4RL data or performance."""
    from arlab.classroom.constants import D4RL_SUITE9
    from arlab.cli import demo_contract
    from arlab.tasks.iql.study import apply_nine_task_protocol
    raw = demo_contract(tmp_path / "seed", steps=2).model_dump(mode="json")
    raw["cases"] = [{"name": name, "split": "dev", "data": str(tmp_path / f"{name}.npz"),
                     "environment": {"kind": "d4rl", "id": name}, "dataset_sha256": "0" * 64}
                    for name in D4RL_SUITE9]
    raw["model"] = config().model_dump(mode="json")
    return apply_nine_task_protocol(raw)


def test_student_session_pins_discovered_model_and_cleans_typed_key(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "_deployment_contract", lambda root: session_contract(tmp_path))
    monkeypatch.delenv("NEW_API_KEY", raising=False)
    monkeypatch.delenv("ARLAB_API_KEY", raising=False)
    monkeypatch.setattr("sys.stdin", type("TTY", (), {"isatty": lambda self: True})())
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("Only key should be requested"))
    monkeypatch.setattr(session.getpass, "getpass", lambda msg: FAKE_KEY)
    opener = StubOpener(model_list("kimi", "kimi-k2.5", "gpt-irrelevant"))
    monkeypatch.setattr(kimi, "build_opener", lambda *args: opener)
    calls = []
    def study(root, contract):
        calls.append(contract)
        assert os.environ["NEW_API_KEY"] == FAKE_KEY
        assert contract.model.model == "kimi"
        return {"status": "fixture_NOT_research", "research_complete": True}
    monkeypatch.setattr(session, "run_study", study)
    assert session.main(["run", "--root", str(tmp_path), "--confirm-budget"]) == 0
    assert len(calls) == 1 and len(opener.requests) == 1
    saved = json.loads((tmp_path / "model.json").read_text())
    assert saved["model"] == "kimi" and FAKE_KEY not in json.dumps(saved)
    assert "NEW_API_KEY" not in os.environ


def test_bad_key_stops_before_research_and_cleans_temporary_key(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "_deployment_contract", lambda root: session_contract(tmp_path))
    monkeypatch.delenv("NEW_API_KEY", raising=False)
    monkeypatch.delenv("ARLAB_API_KEY", raising=False)
    monkeypatch.setattr("sys.stdin", type("TTY", (), {"isatty": lambda self: True})())
    monkeypatch.setattr(session.getpass, "getpass", lambda msg: FAKE_KEY)
    opener = StubOpener(error=HTTPError(LAB_NEW_API_BASE_URL, 401, "no", {}, io.BytesIO(b"")))
    monkeypatch.setattr(kimi, "build_opener", lambda *args: opener)
    monkeypatch.setattr(session, "run_study", lambda *a: pytest.fail("Invalid key must stop before study"))
    assert session.main(["run", "--root", str(tmp_path), "--confirm-budget"]) == 2
    assert "NEW_API_KEY" not in os.environ
    assert not (tmp_path / "runs/study").exists()


def test_resume_keeps_exact_model_even_with_newer_model_available(tmp_path, monkeypatch):
    from arlab.core.io import write_json
    from arlab.core.schema import ResearchContract
    c = session_contract(tmp_path)
    raw = c.model_dump(mode="json")
    raw["model"] = config(model="kimi-k2").model_dump(mode="json")
    pinned = ResearchContract.model_validate(raw)
    write_json(tmp_path / "runs/study/development/contract.json", pinned)
    monkeypatch.setattr(session, "_deployment_contract", lambda root: c)
    monkeypatch.setenv("NEW_API_KEY", FAKE_KEY)
    monkeypatch.setattr("sys.stdin", io.StringIO())
    monkeypatch.setattr(kimi, "build_opener", lambda *args: StubOpener(model_list("kimi-k2", "kimi-k3")))
    calls = []
    def study(root, contract):
        calls.append(contract.model.model)
        return {"status": "fixture_NOT_research", "research_complete": False}
    monkeypatch.setattr(session, "run_study", study)
    assert session.main(["run", "--root", str(tmp_path), "--confirm-budget"]) == 2
    assert calls == ["kimi-k2"]
    assert os.environ["NEW_API_KEY"] == FAKE_KEY  # Owner-provided environment preserved.
