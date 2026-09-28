"""Discover -> acquire -> locate -> cite. Network results never become instructions.

Search results retain metadata/abstract scope. A separate fetched source records
exact bytes, extraction scope, links and retrieval time. Reading creates immutable
span IDs, which hypotheses and research notes may cite alongside trial IDs.
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlencode, urlsplit
from xml.etree import ElementTree
from typing import Any
from arlab.core.io import atomic_write, canonical, digest, read_json
from arlab.core.jobs import LocalBackend
from arlab.core.paging import text_page, list_page
from arlab.core.schema import ResearchAccess
from arlab.core.store import Store
from arlab.research.extract import html_text
from arlab.research.network import PublicHTTP, ResearchError


class ResearchService:
    def __init__(self, root: Path, policy: ResearchAccess, store: Store, *, http: PublicHTTP | None = None,
                 deadline=None):
        self.root, self.policy, self.store = root, policy, store
        self.http = http or PublicHTTP(policy, store, deadline=deadline)

    def _cache(self, key: Any, refresh: bool = False, permanent: bool = False):
        item = self.store.get("research-cache-" + digest(key))
        if not refresh and item and (permanent or time.time() - item["created"] <= self.policy.search_cache_seconds):
            # Cached references are validated, not silently treated as complete if lost.
            return item["result"]
        return None

    def _cache_put(self, key: Any, result: Any):
        self.store.put("research-cache-" + digest(key), {"created": time.time(), "result": result})
        return result

    def _raw(self, body: bytes) -> tuple[str, Path]:
        identity = hashlib.sha256(body).hexdigest()
        path = self.root / "research" / "raw" / identity
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != identity:
                raise ResearchError("Cached research bytes are corrupted")
        else:
            atomic_write(path, body)
        return identity, path

    def _source(self, text: str, metadata: dict) -> str:
        # A text ceiling is explicit; never claim an entire paper was parsed.
        if len(text) > 500_000:
            metadata = {**metadata, "truncated": True, "original_characters": len(text)}
            text = text[:500_000]
        metadata = {"retrieved_at": time.time(), "trust": "external_untrusted",
                    "verified_claim": False, **metadata}
        return self.store.save_artifact("src", text, metadata)

    def search_literature(self, *, query: str, provider="arxiv", offset=0, limit=5, refresh=False) -> dict:
        key = {"tool": "search_literature", "query": query, "provider": provider, "offset": offset, "limit": limit}
        cached = self._cache(key, refresh)
        if cached:
            return {**cached, "cache_hit": True}
        entries = []
        if provider == "arxiv":
            url = "https://export.arxiv.org/api/query?" + urlencode({"search_query": query, "start": offset,
                         "max_results": limit, "sortBy": "relevance", "sortOrder": "descending"})
            response = self.http.get(url)
            raw_hash, _ = self._raw(response.body)
            if b"<!DOCTYPE" in response.body.upper() or b"<!ENTITY" in response.body.upper():
                raise ResearchError("DTD/entity-bearing XML is not accepted")
            tree = ElementTree.fromstring(response.body)
            if tree.tag != "{http://www.w3.org/2005/Atom}feed":
                raise ResearchError("arXiv returned a non-Atom document, not a successful empty search")
            ns = {"a": "http://www.w3.org/2005/Atom", "o": "http://a9.com/-/spec/opensearch/1.1/"}
            total = int(tree.findtext("o:totalResults", default="0", namespaces=ns))
            for node in tree.findall("a:entry", ns):
                def get(tag):
                    return " ".join((node.findtext("a:" + tag, default="", namespaces=ns)).split())
                entry_url = get("id")
                if "api/errors" in entry_url:
                    raise ResearchError("arXiv rejected the query; not an empty result set")
                links = [{"url": e.attrib.get("href", "").replace("http://arxiv.org", "https://arxiv.org"),
                          "type": e.attrib.get("type", "")} for e in node.findall("a:link", ns)]
                entries.append({"title": get("title"), "url": entry_url.replace("http:", "https:", 1),
                                "abstract": get("summary"), "published": get("published"),
                                "updated": get("updated"), "links": links,
                                "authors": [n.findtext("a:name", default="", namespaces=ns) for n in node.findall("a:author", ns)],
                                "scope": "abstract"})
            next_offset = offset + len(entries) if entries and offset + len(entries) < total else None
        elif provider == "crossref":
            url = "https://api.crossref.org/works?" + urlencode({"query.bibliographic": query, "rows": limit, "offset": offset})
            response = self.http.get(url)
            raw_hash, _ = self._raw(response.body)
            body = json.loads(response.body)["message"]
            total = body["total-results"]
            for item in body["items"]:
                abstract = html_text(item.get("abstract", ""))["text"]
                entries.append({"title": " ".join(item.get("title", [])), "url": item.get("URL", ""),
                                "doi": item.get("DOI"), "abstract": abstract,
                                "published": item.get("published", {}), "links": item.get("link", []),
                                "scope": "abstract" if abstract else "metadata"})
            next_offset = offset + len(entries) if entries and offset + len(entries) < total else None
        else:
            raise ValueError("Unsupported literature provider")
        items = []
        for entry in entries:
            text = json.dumps(entry, ensure_ascii=False, indent=2)
            sid = self._source(text, {**entry, "abstract": None, "raw_sha256": raw_hash,
                                      "source_provider": provider, "query": query})
            items.append({"source": sid, "title": entry["title"][:250], "url": entry["url"],
                          "scope": entry["scope"], "snippet": entry.get("abstract", "")[:400]})
        result = {"provider": provider, "items": items, "total_results": total,
                  "next_offset": next_offset, "query": query, "cache_hit": False,
                  "scope": "search_metadata_not_fulltext", "raw_sha256": raw_hash,
                  "warning": "Empty or incomplete search does not establish novelty or absence of prior work"}
        result["search_ref"] = self.store.save_artifact("search", canonical(result), {"provider": provider, "query": query})
        return self._cache_put(key, result)

    def search_web(self, *, query: str, offset=0, limit=5, refresh=False) -> dict:
        if not self.policy.enabled or not self.policy.web_search_enabled:
            raise PermissionError("General web search is disabled; literature/repository research is separate")
        key = {"tool": "search_web", "query": query, "offset": offset, "limit": limit}
        cached = self._cache(key, refresh)
        if cached:
            return {**cached, "cache_hit": True}
        secret = os.environ.get(self.policy.brave_key_env)
        if not secret:
            raise ResearchError(f"Web search is unconfigured; owner must set {self.policy.brave_key_env}. No fallback was fabricated")
        url = "https://api.search.brave.com/res/v1/web/search?" + urlencode({"q": query, "count": limit, "offset": offset})
        response = self.http.get(url, {"X-Subscription-Token": secret, "Accept": "application/json"})
        raw_hash, _ = self._raw(response.body)
        body = json.loads(response.body)
        items = []
        for entry in body.get("web", {}).get("results", [])[:limit]:
            text = entry.get("description", "")
            meta = {"title": entry.get("title", ""), "url": entry["url"], "scope": "search_snippet",
                    "source_provider": "brave", "raw_sha256": raw_hash, "query": query}
            sid = self._source(text, meta)
            items.append({"source": sid, "title": meta["title"], "url": meta["url"], "snippet": text[:400],
                          "scope": "search_snippet"})
        result = {"items": items, "provider": "brave", "query": query, "cache_hit": False,
                  "next_offset": offset + 1 if body.get("query", {}).get("more_results_available") and offset < 9 else None}
        result["search_ref"] = self.store.save_artifact("search", canonical(result), {"provider": "brave"})
        return self._cache_put(key, result)

    def fetch_source(self, *, url: str, refresh=False, extra_metadata: dict | None = None) -> dict:
        key = {"tool": "fetch_source", "url": url, "extra": extra_metadata}
        cached = self._cache(key, refresh, permanent=True)
        if cached:
            if not self.store.artifact(cached["source"]):
                raise ResearchError("Source cache reference is missing")
            return {**cached, "cache_hit": True}
        response = self.http.get(url)
        raw_hash, raw_path = self._raw(response.body)
        content_type = response.headers.get("content-type", "").split(";")[0].lower()
        if response.body.startswith(b"%PDF-"):
            import importlib.util
            if importlib.util.find_spec("pypdf") is None:
                raise ResearchError("PDF text reading requires owner installation: pip install '.[research]' ; raw bytes retained")
            parsed = self.root / "research" / "parsed" / (raw_hash + ".json")
            parsed.parent.mkdir(parents=True, exist_ok=True)
            task = LocalBackend().run([sys.executable, "-m", "arlab.research.parse_worker", str(raw_path), str(parsed)],
                                     parsed.parent, parsed.with_suffix(".log"), min(25, self.http.remaining()))
            if task.exit_code:
                raise ResearchError("PDF text parser failed/timed out; no reconstructed text or OCR fallback")
            extraction = read_json(parsed)
        elif content_type in {"text/html", "application/xhtml+xml"} or response.body.lstrip().lower().startswith((b"<!doctype html", b"<html")):
            extraction = html_text(response.body.decode("utf-8", errors="replace"), response.url)
        elif content_type.startswith("text/") or content_type in {"application/json", "application/xml", "application/atom+xml", ""}:
            if b"\0" in response.body[:10000]:
                raise ResearchError("Binary research content is unsupported")
            extraction = {"text": response.body.decode("utf-8", errors="replace"), "scope": "text"}
        else:
            raise ResearchError(f"Unsupported research content type: {content_type}")
        text = extraction.pop("text")
        if not text.strip():
            raise ResearchError("No readable source text; not a successful full-paper reading")
        metadata = {"url": response.url, "requested_url": url, "title": extraction.pop("title", "") or response.url,
                    "raw_sha256": raw_hash, "content_type": content_type, **extraction, **(extra_metadata or {})}
        sid = self._source(text, metadata)
        result = {"source": sid, "title": metadata["title"], "url": response.url,
                  "scope": metadata["scope"], "total_lines": len(self.store.artifact(sid)["text"].splitlines()),
                  "characters": min(len(text), 500_000), "raw_sha256": raw_hash,
                  "truncated": len(text) > 500_000 or metadata.get("truncated", False),
                  "preview": text[:700], "read_more": "read_source", "cache_hit": False}
        if metadata.get("page_map"):
            result["pages_extracted"] = metadata["pages_extracted"]
            result["visuals_verified"] = False
        return self._cache_put(key, result)

    def read_source(self, *, source: str, start_line=1, max_lines=60, max_characters=5000, offset=0) -> dict:
        artifact = self.store.artifact(source)
        if not artifact or artifact["kind"] not in ("src", "span"):
            raise ValueError("Unknown readable source or span")
        page = text_page(artifact["text"], start_line, max_lines, max_characters, offset)
        meta = artifact["metadata"]
        pages = [p["page"] for p in meta.get("page_map", [])
                 if p["start_line"] <= page["end_line"] and p["end_line"] >= start_line]
        span = self.store.save_artifact("span", page["text"], {
            "source": source, "start_line": start_line, "end_line": page["end_line"],
            "offset": offset, "next": page["next"], "scope": meta.get("scope"),
            "raw_sha256": meta.get("raw_sha256"), "trust": "external_untrusted",
            "url": meta.get("url"), "title": meta.get("title"), "pdf_pages": pages,
        })
        return {"source": source, "span": span, "scope": meta.get("scope"), "url": meta.get("url"),
                "title": meta.get("title"), "trust": "external_untrusted", "pdf_pages": pages, **page}

    def find_source(self, *, source: str, query: str, offset=0, limit=10) -> dict:
        artifact = self.store.artifact(source)
        if not artifact or artifact["kind"] not in ("src", "span"):
            raise ValueError("Unknown source")
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        matches = []
        for i, line in enumerate(artifact["text"].splitlines(), 1):
            match = pattern.search(line)
            if match:
                start = max(0, match.start() - 80)
                matches.append({"line": i, "offset": start, "match_offset": match.start(),
                                "preview": line[start:start + 300]})
        return {"source": source, **list_page(matches, offset, limit)}

    def _github(self, path: str) -> dict | list:
        headers = {"Accept": "application/vnd.github+json"}
        secret = os.environ.get(self.policy.github_token_env)
        if secret:
            headers["Authorization"] = "Bearer " + secret
        response = self.http.get("https://api.github.com" + path, headers)
        self._raw(response.body)
        return json.loads(response.body)

    def search_repositories(self, *, query: str, page=1, limit=5, refresh=False) -> dict:
        key = {"tool": "search_repositories", "query": query, "page": page, "limit": limit}
        cached = self._cache(key, refresh)
        if cached:
            return {**cached, "cache_hit": True}
        body = self._github("/search/repositories?" + urlencode({"q": query + " is:public", "page": page, "per_page": limit}))
        items = []
        for r in body["items"]:
            meta = {"title": r["full_name"], "url": r["html_url"], "scope": "repository_search_metadata"}
            sid = self._source(json.dumps({"repository": r["full_name"], "description": r.get("description"),
                           "license": r.get("license"), "default_branch": r["default_branch"]}, indent=2), meta)
            items.append({"source": sid, "repository": r["full_name"], "description": r.get("description"), "url": r["html_url"]})
        result = {"items": items, "total_results": body["total_count"], "cache_hit": False,
                  "incomplete_results": body.get("incomplete_results", False),
                  "next_page": page + 1 if page * limit < min(body["total_count"], 1000) else None,
                  "warning": "Repository name/ranking is not proof of author-official status; inspect provenance and license"}
        return self._cache_put(key, result)

    def read_repository(self, *, repository: str, ref="HEAD", path="", offset=0, limit=40) -> dict:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise ValueError("Expected owner/repository")
        if any(p in (".", "..") for p in repository.split("/")):
            raise ValueError("Invalid repository name")
        if path and ("\\" in path or PurePosixPath(path).is_absolute() or
                     any(p in (".", "..", "") for p in path.split("/")) or any(ord(c) < 32 for c in path)):
            raise ValueError("Invalid repository path")
        pin_key = {"repository": repository, "ref": ref, "type": "immutable_pin"}
        pin = self._cache(pin_key, permanent=True)
        if not pin:
            info = self._github("/repos/" + repository)
            if info.get("private", False):
                raise PermissionError("This research adapter reads public repositories only")
            chosen = info["default_branch"] if ref == "HEAD" else ref
            commit = self._github("/repos/" + repository + "/commits/" + quote(chosen, safe=""))
            sha = commit["sha"]
            if not re.fullmatch(r"[0-9a-f]{40}", sha):
                raise ResearchError("Invalid immutable Git commit")
            pin = self._cache_put(pin_key, {"commit": sha, "repository": repository,
                    "license": info.get("license"), "requested_ref": ref})
        sha = pin["commit"]
        self._cache_put({"repository": repository, "ref": sha, "type": "immutable_pin"}, pin)
        if path:
            fetched = self.fetch_source(url=f"https://raw.githubusercontent.com/{repository}/{sha}/{quote(path, safe='/')}",
                extra_metadata={"repository": repository, "commit": sha, "path": path,
                                "license": pin["license"], "scope": "repository_file", "title": repository + ":" + path})
            return {**fetched, "commit": sha, "repository": repository, "path": path}
        tree_key = {"tool": "repository_tree", "repository": repository, "commit": sha}
        tree = self._cache(tree_key, permanent=True)
        if not tree:
            body = self._github(f"/repos/{repository}/git/trees/{sha}?recursive=1")
            entries = [{"path": e["path"], "type": e["type"], "size": e.get("size"), "sha": e["sha"]} for e in body["tree"]]
            tree = self._cache_put(tree_key, {"entries": entries, "truncated_by_provider": body.get("truncated", False)})
        return {"repository": repository, "commit": sha, "license": pin["license"],
                "truncated_by_provider": tree["truncated_by_provider"], **list_page(tree["entries"], offset, limit),
                "next_read": "read_repository(repository, ref=commit, path=selected_file)"}
