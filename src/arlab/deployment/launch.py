"""Owner launch gate: installation is not permission to spend a research budget."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from arlab.cli import load_contract
from arlab.core.endpoints import AUTO_KIMI, is_lab_gateway, validate_model_endpoint
from arlab.core.io import file_digest, read_json, source_digest
from arlab.core.schema import ResearchContract


def launch_plan(root: Path, args: argparse.Namespace) -> tuple[ResearchContract, dict]:
    state = read_json(root / "deployment.json")
    if state["status"] != "ready":
        raise ValueError("Deployment has not passed training/evaluation smoke; complete deploy.sh first")
    check = read_json(root / "checks.json")
    if check["source_sha256"] != source_digest():
        raise ValueError("Implementation changed after deployment validation; deploy into a new root")
    raw = load_contract(root / "configs/research.json").model_dump(mode="json")
    for name in ("provider", "model", "base_url"):
        value = getattr(args, name, None)
        if value:
            raw["model"][name] = value
    if getattr(args, "research_online", False):
        raw["research_access"]["enabled"] = True
        raw["research_access"]["allowed_hosts"] = list(dict.fromkeys(
            raw["research_access"]["allowed_hosts"] + getattr(args, "research_host", [])))
    elif getattr(args, "research_host", []):
        raise ValueError("--research-host requires --research-online")
    contract = ResearchContract.model_validate(raw)
    for case in contract.cases:
        if not case.dataset_sha256 or file_digest(Path(case.data)) != case.dataset_sha256:
            raise ValueError("Dataset missing or differs from the deployment contract")
    return contract, {"name": contract.name, "cases": len(contract.cases), "device": contract.device,
                      "rungs": contract.rungs, "training_seeds": contract.train_seeds,
                      "code_mode": contract.code_execution.mode, "budget": contract.budget.model_dump(),
                      "provider": contract.model.provider, "model": contract.model.model,
                      "research_access": contract.research_access.model_dump(), "calls_llm_now": False}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--provider", choices=("compatible", "responses", "anthropic"))
    p.add_argument("--model")
    p.add_argument("--base-url")
    p.add_argument("--research-online", action="store_true")
    p.add_argument("--research-host", action="append", default=[])
    p.add_argument("--confirm-budget", action="store_true")
    p.add_argument("--acknowledge-local-code", action="store_true")
    p.add_argument("--allow-no-api-key", action="store_true", help="Only for a loopback local model server")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    try:
        contract, summary = launch_plan(args.root, args)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if args.dry_run:
            return 0
        if not args.confirm_budget:
            raise ValueError("Review the displayed budget, then explicitly pass --confirm-budget")
        if "REPLACE" in contract.model.model or not contract.model.model.strip():
            raise ValueError("Configure a real model name; a successful install is not an LLM research run")
        local = validate_model_endpoint(contract.model.base_url,
                                        allow_lab_http=contract.model.allow_lab_http)
        if not os.environ.get(contract.model.api_key_env) and not (local and args.allow_no_api_key):
            raise ValueError(f"Set {contract.model.api_key_env}; the launcher never saves its value")
        if os.environ.get("ARLAB_LOCAL_ROOT") and contract.code_execution.mode == "docker":
            raise ValueError("Docker is disabled in this project-local uv installation")
        if contract.code_execution.mode == "trusted_local" and os.environ.get("ARLAB_LOCAL_ROOT"):
            from arlab.deployment.local import LOCAL_CODE_NOTICE
            print(LOCAL_CODE_NOTICE, flush=True)
        if contract.code_execution.mode == "trusted_local" and not args.acknowledge_local_code and not os.environ.get("ARLAB_LOCAL_ROOT"):
            raise ValueError("Local Python execution is not sandboxed; review code and pass --acknowledge-local-code")
        # Resolve from the old contract BEFORE discovery: an auto preset must
        # never select a newer API model while resuming the same study.
        previous = (args.output or args.root / ("runs/study" if contract.final_evaluation else "runs/research"))
        previous_contract = previous / ("development/contract.json" if contract.final_evaluation else "contract.json")
        if previous_contract.exists():
            pinned = ResearchContract.model_validate(read_json(previous_contract))
            same_endpoint = (contract.model.provider, contract.model.base_url) == (pinned.model.provider, pinned.model.base_url)
            if contract.model.model == AUTO_KIMI and same_endpoint:
                raw = contract.model_dump(mode="json")
                raw["model"] = pinned.model.model_dump(mode="json")
                contract = ResearchContract.model_validate(raw)
            elif pinned.model != contract.model:
                raise ValueError("Model differs from the existing campaign; use a new study")
        from arlab.classroom.kimi import HTTP_NOTICE, prepare_model
        if is_lab_gateway(contract.model.base_url):
            print(HTTP_NOTICE, flush=True)
        resolved = prepare_model(contract.model)
        if resolved != contract.model:
            raw = contract.model_dump(mode="json")
            raw["model"] = resolved.model_dump(mode="json")
            contract = ResearchContract.model_validate(raw)
        if contract.final_evaluation:
            from arlab.classroom.session import run_study, budget_preview
            print(json.dumps(budget_preview(contract), ensure_ascii=False, indent=2))
            result = run_study(args.output or args.root / "runs/study", contract)
            return 0 if result["research_complete"] else 2
        from arlab.core.harness import Harness
        output = args.output or args.root / "runs/research"
        harness = Harness(contract, output)
        try:
            result = harness.run()
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["status"] == "completed" else 2
        finally:
            harness.close()
    except Exception as exc:
        print(f"LAUNCH ERROR {type(exc).__name__}: {exc}", file=__import__("sys").stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
