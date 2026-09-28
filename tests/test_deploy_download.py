"""Transport fault injection. These tests do NOT assert live server availability."""
from __future__ import annotations

import hashlib
import io
import urllib.error

import pytest

from arlab.deployment.archive import extract_mujoco
from arlab.deployment.common import exclusive_lock, read_json
from arlab.deployment.download import DownloadError, fetch_https, validate_url

URL = "https://data.example.org/sample.hdf5"


class Response:
    def __init__(self, data=b"abcdefgh", status=200, headers=None, fail_after=None):
        self.stream = io.BytesIO(data)
        self.status = status
        self.headers = headers or {"Content-Length": str(len(data)), "ETag": '"v1"'}
        self.fail_after = fail_after
        self.reads = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stream.close()

    def read(self, size):
        self.reads += 1
        if self.fail_after and self.reads > 1:
            raise ConnectionResetError("injected disconnect")
        return self.stream.read(self.fail_after or size)


def transfer(tmp_path, responses, **kwargs):
    calls = []
    def opener(url, headers, hosts, timeout):
        calls.append(dict(headers))
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result
    dest = tmp_path / "file"
    result = fetch_https(URL, dest, opener=opener, sleep=lambda _: None, **kwargs)
    return dest, result, calls


def test_atomic_download_receipt_and_verified_reuse(tmp_path):
    dest, result, _ = transfer(tmp_path, [Response()])
    assert dest.read_bytes() == b"abcdefgh"
    assert result == hashlib.sha256(b"abcdefgh").hexdigest()
    receipt = read_json(dest.with_name("file.download.json"))
    assert receipt["integrity"] == "https_and_local_identity"
    fetch_https(URL, dest, opener=lambda *_: pytest.fail("cache should not download"))
    dest.write_bytes(b"tampered")
    with pytest.raises(DownloadError, match="receipt"):
        fetch_https(URL, dest)


def test_resume_uses_if_range_and_validates_entity(tmp_path):
    responses = [Response(fail_after=3), Response(b"defgh", 206,
                 {"Content-Length": "5", "Content-Range": "bytes 3-7/8", "ETag": '"v1"'})]
    dest, _, calls = transfer(tmp_path, responses)
    assert dest.read_bytes() == b"abcdefgh"
    assert calls[1]["Range"] == "bytes=3-"
    assert calls[1]["If-Range"] == '"v1"'


def test_server_ignoring_range_restarts_instead_of_appending(tmp_path):
    dest, _, calls = transfer(tmp_path, [Response(fail_after=3), Response(b"new entity")])
    assert "Range" in calls[1]
    assert dest.read_bytes() == b"new entity"


def test_no_validator_means_no_resume_without_expected_hash(tmp_path):
    dest, _, calls = transfer(tmp_path, [Response(headers={"Content-Length": "8"}, fail_after=3), Response()])
    assert "Range" not in calls[1]
    assert dest.read_bytes() == b"abcdefgh"


@pytest.mark.parametrize("headers", [
    {"Content-Range": "bytes 4-7/8", "ETag": '"v1"'},
    {"Content-Range": "bytes 3-7/8", "ETag": '"v2"'},
    {"Content-Range": "bytes 3-6/8", "ETag": '"v1"'},
    {"Content-Range": "bytes 3-7/8", "ETag": '"v1"', "Content-Length": "6"},
])
def test_bad_range_or_changed_validator_never_publishes(tmp_path, headers):
    with pytest.raises(DownloadError):
        transfer(tmp_path, [Response(fail_after=3), Response(b"defgh", 206, headers)])
    assert not (tmp_path / "file").exists()


def test_wrong_expected_hash_discards_partial(tmp_path):
    with pytest.raises(DownloadError, match="checksum"):
        transfer(tmp_path, [Response()], expected_sha256="0" * 64)
    assert not (tmp_path / "file").exists()
    assert not (tmp_path / "file.part").exists()


@pytest.mark.parametrize("url", ["http://data.example.org/f", "https://evil.example/f",
                                  "https://user:secret@data.example.org/f", "https://data.example.org:444/f"])
def test_url_security_policy(url):
    with pytest.raises(DownloadError):
        validate_url(url, {"data.example.org"})


def test_size_limit_does_not_publish(tmp_path):
    with pytest.raises(DownloadError, match="limit"):
        transfer(tmp_path, [Response()], max_bytes=3)
    assert not (tmp_path / "file").exists()


def test_encoded_response_rejected(tmp_path):
    with pytest.raises(DownloadError, match="Encoded"):
        transfer(tmp_path, [Response(headers={"Content-Encoding": "gzip"})])


def test_retry_transient_but_not_404(tmp_path):
    transient = urllib.error.HTTPError(URL, 503, "busy", {}, None)
    _, _, calls = transfer(tmp_path, [transient, Response()])
    assert len(calls) == 2
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(DownloadError, match="404"):
        transfer(other, [urllib.error.HTTPError(URL, 404, "missing", {}, None)])


def test_416_restarts_without_trusting_remote_size(tmp_path):
    _, _, calls = transfer(tmp_path, [Response(fail_after=3),
        urllib.error.HTTPError(URL, 416, "range", {}, None), Response()])
    assert "Range" not in calls[2]


def test_truncation_retries_then_fails_without_final_file(tmp_path):
    with pytest.raises(DownloadError, match="2 attempts"):
        transfer(tmp_path, [Response(b"a", headers={"Content-Length": "20"}),
                            Response(b"a", headers={"Content-Length": "20"})], attempts=2)
    assert not (tmp_path / "file").exists()


def test_symlink_destination_rejected(tmp_path):
    (tmp_path / "target").write_text("original")
    (tmp_path / "file").symlink_to(tmp_path / "target")
    with pytest.raises(DownloadError, match="symlink"):
        transfer(tmp_path, [Response()])
    assert (tmp_path / "target").read_text() == "original"


def test_lock_excludes_concurrent_writers(tmp_path):
    with exclusive_lock(tmp_path / "lock"):
        with pytest.raises(RuntimeError, match="Another"):
            with exclusive_lock(tmp_path / "lock"):
                pass


def make_tar(path, entries):
    import tarfile
    with tarfile.open(path, "w:gz") as tar:
        for name, payload, link in entries:
            member = tarfile.TarInfo(name)
            if link:
                member.type = tarfile.SYMTYPE
                member.linkname = link
                tar.addfile(member)
            else:
                member.size = len(payload)
                tar.addfile(member, io.BytesIO(payload))


def test_safe_archive_and_internal_link(tmp_path):
    archive = tmp_path / "archive.tar.gz"
    make_tar(archive, [("mujoco210/bin/libmujoco210.so", b"test-library", None),
                       ("mujoco210/bin/alias.so", b"", "libmujoco210.so")])
    extract_mujoco(archive, tmp_path / "out")
    assert (tmp_path / "out/mujoco210/bin/alias.so").read_bytes() == b"test-library"


@pytest.mark.parametrize("name,link", [("../outside", None), ("/absolute", None),
                                       ("mujoco210/link", "../../outside")])
def test_archive_traversal_and_escaping_links_rejected(tmp_path, name, link):
    archive = tmp_path / "archive.tar.gz"
    make_tar(archive, [(name, b"bad", link)])
    with pytest.raises(ValueError):
        extract_mujoco(archive, tmp_path / "out")
    assert not (tmp_path / "outside").exists()
