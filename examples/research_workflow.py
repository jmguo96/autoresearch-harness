"""A visible research-tool walkthrough followed by REAL IQL/candidate experiments.

Decision order is scripted, not LLM autonomy. --offline-fixture uses labelled HTTP
fixtures, never a silent network fallback. Without it, sources are fetched live.
Use `arlab run ... --provider compatible --research-online` for a real LLM agent.
"""
from __future__ import annotations
import argparse
import contextlib
import json
import socket
from pathlib import Path
from arlab.cli import demo_contract
from arlab.core.harness import Harness
from arlab.core.io import write_json
from arlab.core.models import ModelReply
from arlab.core.teaching import TeachingModel
from arlab.research.network import PublicHTTP, Response
from arlab.research.service import ResearchService


class Walkthrough(TeachingModel):
    def _research(self, context):
        counts = context["teaching"]["action_counts"]
        last = context.get("last_feedback") or {}
        details = last.get("details", {})
        def decision(action):
            return {"summary": "Scripted research-tool walkthrough, not an autonomous discovery", "action": action}
        if not counts.get("search_literature"):
            return decision({"kind": "search_literature", "query": 'ti:"Offline Reinforcement Learning with Implicit Q-Learning"', "limit": 1})
        if not counts.get("fetch_source"):
            return decision({"kind": "fetch_source", "url": "https://arxiv.org/html/2110.06169v1"})
        if not counts.get("read_source"):
            return decision({"kind": "read_source", "source": details["source"], "max_lines": 20, "max_characters": 2500})
        if not counts.get("remember"):
            return decision({"kind": "remember", "category": "literature", "evidence_ids": [details["span"]],
                             "text": "IQL research source passage was retrieved. Its contents do not establish any candidate improvement; test the hypothesis separately."})
        if not counts.get("read_repository"):
            return decision({"kind": "read_repository", "repository": "corl-team/CORL", "path": "algorithms/offline/iql.py"})
        children = [c for c in context["candidates"] if c["id"] != context["baseline"]]
        if children and not counts.get("check_candidate"):
            return decision({"kind": "check_candidate", "candidate": children[-1]["id"],
                             "case": context["protocol"]["dev_cases"][0], "train_seed": 0, "steps": 2})
        result = super()._research(context)
        if result["action"]["kind"] == "propose":
            result["action"]["hypothesis"]["evidence_ids"] = [e for m in context["memory"] for e in m["evidence_ids"]][:5]
        return result


def fixture_http(policy, store, root):
    def resolver(*args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))]
    def exchange(url, headers, timeout, maximum):
        if "/api/query?" in url:
            body = b'''<feed xmlns="http://www.w3.org/2005/Atom" xmlns:o="http://a9.com/-/spec/opensearch/1.1/"><o:totalResults>1</o:totalResults><entry><id>https://arxiv.org/abs/2110.06169v1</id><title>OFFLINE FIXTURE: IQL source</title><summary>Not downloaded from arXiv. This is a test of the research tool flow.</summary></entry></feed>'''
            content = "application/atom+xml"
        elif "/html/" in url:
            body = b"<html><title>OFFLINE FIXTURE, not paper text</title><p>IQL research-tool integration test.</p><p>No scientific finding is supplied by this fixture.</p></html>"
            content = "text/html"
        elif "/commits/" in url:
            body = json.dumps({"sha": "a" * 40}).encode(); content = "application/json"
        elif url.startswith("https://api.github.com/repos/"):
            body = json.dumps({"private": False, "default_branch": "main", "license": {"spdx_id": "NOASSERTION"}}).encode()
            content = "application/json"
        elif url.startswith("https://raw.githubusercontent.com/"):
            body = b"# OFFLINE FIXTURE: this is not downloaded CORL code and is never executed.\n"; content = "text/plain"
        else:
            raise AssertionError("Unconfigured fixture URL " + url)
        return Response(url, 200, {"content-type": content}, body)
    http = PublicHTTP(policy, store, resolver=resolver, exchange=exchange, limiter_dir=root / "rate")
    http._rate_limit = lambda host: contextlib.nullcontext()
    return http


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--offline-fixture", action="store_true")
    args = parser.parse_args()
    contract = demo_contract(args.output, steps=args.steps, candidates=2, open_code=True)
    contract.research_access.enabled = True
    contract.model.model = "SCRIPTED-research-tool-walkthrough"
    agent = Harness(contract, args.output / "campaign")
    agent.model = Walkthrough(agent.task, contract.seed_recipe, True)
    if args.offline_fixture:
        agent.research = ResearchService(agent.root, contract.research_access, agent.store,
            http=fixture_http(contract.research_access, agent.store, args.output))
    try:
        report = agent.run()
        report["walkthrough"] = {"decisions": "scripted", "external_sources": "offline_fixtures" if args.offline_fixture else "live_network",
                                 "training_and_rollout": "real_synthetic_environment", "d4rl_performance_claim": False}
        write_json(args.output / "walkthrough-report.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "completed" else 2
    finally:
        agent.close()


if __name__ == "__main__":
    raise SystemExit(main())
