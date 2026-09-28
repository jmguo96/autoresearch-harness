"""Explicit commands for researchers and owners; no hidden background service."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

from arlab.core.io import canonical, directory_digest, file_digest, read_json, source_digest, write_json
from arlab.core.schema import Budget, Case, CodeExecutionPolicy, ModelConfig, ResearchContract
from arlab.core.config import AgentConfig


def load_contract(path: Path) -> ResearchContract:
    with path.open(encoding="utf-8") as stream:
        raw = json.load(stream) if path.suffix == ".json" else yaml.safe_load(stream)
    for case in raw["cases"]:
        data = Path(case["data"]).expanduser()
        case["data"] = str((path.parent / data).resolve() if not data.is_absolute() else data.resolve())
    return ResearchContract.model_validate(raw)


def demo_contract(root: Path, *, steps: int = 80, candidates: int = 3,
                  name: str = "iql-lesson", dynamics_offset: int = 0, open_code: bool = False) -> ResearchContract:
    from arlab.tasks.iql.environments import make_fixture
    from arlab.tasks.iql.recipe import IQLRecipe
    root = root.resolve()
    cases = []
    for index, split in enumerate(("dev", "validation", "test")):
        path = root / "data" / f"{split}.npz"
        dynamics = dynamics_offset + index * 17
        if not path.exists():
            make_fixture(path, dynamics_seed=dynamics, data_seed=31 + index, episodes=48, horizon=16)
        cases.append(Case(name=f"lq-{split}", split=split, data=str(path), dataset_sha256=file_digest(path),
                          environment={"kind": "synthetic_lq", "dynamics_seed": dynamics, "horizon": 16}))
    return ResearchContract(name=name, goal="Research algorithms with in-sample value learning; use unchanged IQL as the baseline. Networks, objectives, policies, and update schedules may change.",
        code_execution=CodeExecutionPolicy(mode="trusted_local" if open_code else "disabled"),
        cases=cases, seed_recipe=IQLRecipe(hidden_dims=[32, 32], batch_size=64).model_dump(mode="json"),
        train_seeds=[0], eval_seeds=[1000, 1001, 1002], rungs=[steps], max_candidates=candidates,
        budget=Budget(max_decisions=30, max_trials=candidates + 2, max_train_steps=(candidates + 2) * steps,
                      max_eval_episodes=(candidates + 2) * 3, max_wall_seconds=1800, worker_timeout_seconds=180))


def _run(contract: ResearchContract, output: Path, config: AgentConfig | None = None,
         docker_image: str | None = None) -> dict[str, Any]:
    from arlab.core.harness import Harness
    agent = Harness(contract, output, config=config, docker_image=docker_image)
    try:
        return agent.run()
    finally:
        agent.close()


def _print(result: Any) -> None:
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arlab", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("demo", help="Run real IQL training with a scripted teaching controller")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--candidates", type=int, default=3)
    p = commands.add_parser("open-demo", help="Run reviewed open-code candidates and fixed IQL on synthetic data")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--candidates", type=int, default=3)
    p = commands.add_parser("template", help="Export an editable algorithm SOURCE bundle")
    p.add_argument("--name", choices=["ensemble-mixture", "diffusion-extraction"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source-dir", type=Path)
    p.add_argument("--tiny", action="store_true")
    p = commands.add_parser("bundle", help="Read edited Python files back into an immutable candidate recipe")
    p.add_argument("--recipe", type=Path, required=True)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("run", help="Run an autonomous research campaign")
    p.add_argument("config", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--agent-config", type=Path)
    p.add_argument("--research-online", action="store_true", help="Owner authorizes public research requests")
    p.add_argument("--research-host", action="append", default=[], help="Additional exact HTTPS host grants")
    p.add_argument("--docker-image")
    p.add_argument("--provider", choices=["demo", "compatible", "responses", "anthropic"])
    p.add_argument("--model")
    p.add_argument("--base-url")
    p.add_argument("--code-mode", choices=["disabled", "trusted_local", "docker"])
    p.add_argument("--code-image")
    p = commands.add_parser("resume")
    p.add_argument("campaign", type=Path)
    p.add_argument("--docker-image")
    p = commands.add_parser("status")
    p.add_argument("campaign", type=Path)
    p = commands.add_parser("prepare-d4rl")
    p.add_argument("--env", required=True)
    p.add_argument("--hdf5", type=Path, help="Existing raw D4RL-v2 HDF5; download only when omitted")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--raw-sha256", default="")
    p = commands.add_parser("download-d4rl", help="Batch HTTPS download/import with verified resume and conversion receipts")
    select = p.add_mutually_exclusive_group()
    select.add_argument("--env", default="hopper-medium-v2")
    select.add_argument("--suite12", action="store_true")
    select.add_argument("--suite9", action="store_true")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--raw-dir", type=Path)
    p.add_argument("--checksums", type=Path)
    p.add_argument("--offline", action="store_true")
    p = commands.add_parser("init-d4rl")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--env", default="hopper-medium-v2")
    suite = p.add_mutually_exclusive_group()
    suite.add_argument("--suite12", action="store_true")
    suite.add_argument("--suite9", action="store_true")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--model", default="REPLACE_WITH_YOUR_MODEL")
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--code-mode", choices=["disabled", "trusted_local", "docker"], default="disabled")
    p.add_argument("--code-image")
    p = commands.add_parser("train")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--recipe", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=1000000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--code-mode", choices=["disabled", "trusted_local", "docker"], default="disabled")
    p.add_argument("--code-image")
    p = commands.add_parser("evaluate")
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--env", required=True, help="A legacy D4RL-v2 ID, or synthetic_lq")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=9000)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--dynamics-seed", type=int, default=0)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--code-mode", choices=["disabled", "trusted_local", "docker"], default="disabled")
    p.add_argument("--code-image")
    p = commands.add_parser("audit")
    p.add_argument("campaign", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=["validation", "test", "final"], default="test")
    p.add_argument("--train-seeds", nargs="+", type=int)
    p.add_argument("--eval-seeds", nargs="+", type=int)
    p.add_argument("--optimizer-step-limit", type=int)
    p.add_argument("--dry-run", action="store_true")
    p = commands.add_parser("research", help="Owner-run research tool on an existing campaign")
    p.add_argument("campaign", type=Path)
    p.add_argument("--action", type=Path, required=True, help="JSON research action; online access must already be granted")
    p = commands.add_parser("import-source", help="Owner imports a local text document; never an agent filesystem tool")
    p.add_argument("campaign", type=Path)
    p.add_argument("path", type=Path)
    p.add_argument("--title", default="")
    p.add_argument("--url", default="")
    p = commands.add_parser("context", help="Show the next bounded input projection without calling a model")
    p.add_argument("campaign", type=Path)
    p = commands.add_parser("export")
    p.add_argument("campaign", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("doctor")
    p.add_argument("--d4rl-env")
    p = commands.add_parser("classroom", help="Student entry; normal users use bash start.sh")
    p.add_argument("arguments", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "classroom":
            from arlab.classroom.session import main as classroom_main
            return classroom_main(args.arguments)
        if args.command in ("demo", "open-demo"):
            contract = demo_contract(args.output, steps=args.steps, candidates=args.candidates,
                                     open_code=args.command == "open-demo")
            result = _run(contract, args.output / "campaign")
            _print(result)
            if result["status"] != "completed":
                return 2
        elif args.command == "template":
            from arlab.algorithms.templates import template
            from arlab.algorithms.spec import CodeRecipe
            recipe = CodeRecipe.model_validate(template(args.name, tiny=args.tiny))
            write_json(args.output, recipe)
            if args.source_dir:
                recipe.materialize(args.source_dir)
            _print({"recipe": str(args.output.resolve()), "source_sha256": recipe.source_sha256})
        elif args.command == "bundle":
            from arlab.algorithms.spec import CodeRecipe
            raw = read_json(args.recipe)
            source = args.source_dir.resolve()
            paths = sorted(source.rglob("*.py"))
            if any(p.is_symlink() or not p.resolve().is_relative_to(source) for p in paths):
                raise ValueError("Source bundle cannot contain symlinks/escapes")
            raw["files"] = {str(p.relative_to(source)): p.read_text(encoding="utf-8") for p in paths}
            recipe = CodeRecipe.model_validate(raw)
            write_json(args.output, recipe)
            _print({"recipe": str(args.output.resolve()), "source_sha256": recipe.source_sha256})
        elif args.command == "run":
            contract = load_contract(args.config)
            raw = contract.model_dump(mode="json")
            for key in ("provider", "model", "base_url"):
                value = getattr(args, key)
                if value is not None:
                    raw["model"][key] = value
            if args.code_mode is not None:
                raw["code_execution"]["mode"] = args.code_mode
            if args.code_image is not None:
                raw["code_execution"]["docker_image"] = args.code_image
            contract = ResearchContract.model_validate(raw)
            if args.research_online:
                raw = contract.model_dump(mode="json")
                raw["research_access"]["enabled"] = True
                raw["research_access"]["allowed_hosts"] = list(dict.fromkeys(raw["research_access"]["allowed_hosts"] + args.research_host))
                contract = ResearchContract.model_validate(raw)
            elif args.research_host:
                raise ValueError("--research-host requires --research-online")
            agent_config = AgentConfig.model_validate(read_json(args.agent_config)) if args.agent_config else None
            result = _run(contract, args.output, agent_config, args.docker_image)
            _print(result)
            if result["status"] != "completed":
                return 2
        elif args.command == "resume":
            contract = ResearchContract.model_validate(read_json(args.campaign / "contract.json"))
            config = AgentConfig.model_validate(read_json(args.campaign / "agent.json"))
            result = _run(contract, args.campaign, config, args.docker_image)
            _print(result)
            if result["status"] != "completed":
                return 2
        elif args.command == "status":
            _print(read_json(args.campaign / "report.json"))
        elif args.command == "prepare-d4rl":
            from arlab.deployment.tasks import prepare_one
            _print(prepare_one(args.env, args.output, raw=args.hdf5,
                               expected_sha256=args.raw_sha256, offline=args.hdf5 is not None))
        elif args.command == "download-d4rl":
            from arlab.deployment.tasks import prepare_many
            from arlab.tasks.iql.environments import D4RL_IDS
            from arlab.classroom.constants import D4RL_SUITE9
            _print(prepare_many(args.data_root, list(D4RL_SUITE9) if args.suite9 else list(D4RL_IDS) if args.suite12 else [args.env],
                                raw_dir=args.raw_dir, offline=args.offline,
                                checksums=read_json(args.checksums) if args.checksums else None))
        elif args.command == "init-d4rl":
            from arlab.tasks.iql.environments import D4RL_IDS, reference
            from arlab.tasks.iql.recipe import IQLRecipe
            from arlab.core.presets import adaptive_exploration, research_budget
            from arlab.classroom.constants import D4RL_SUITE9
            ids = D4RL_SUITE9 if args.suite9 else D4RL_IDS if args.suite12 else (args.env,)
            cases = []
            for env_id in ids:
                reference(env_id)
                split = "test" if args.suite12 and "medium-expert" in env_id else (
                    "validation" if args.suite12 and "medium-replay" in env_id else "dev")
                path = (args.data_root / f"{env_id}.npz").resolve()
                cases.append(Case(name=env_id, split=split, data=str(path), environment={"kind": "d4rl", "id": env_id},
                                  dataset_sha256=file_digest(path) if path.exists() else ""))
            contract = ResearchContract(name="d4rl-iql-insample", goal="Research in-sample IQL variants with matched experiments; preserve the unchanged IQL baseline.",
                cases=cases, seed_recipe=IQLRecipe().model_dump(mode="json"), train_seeds=[0, 1, 2],
                code_execution=CodeExecutionPolicy(mode=args.code_mode, docker_image=args.code_image, container_gpu=args.device == "cuda"),
                eval_seeds=list(range(1000, 1010)), rungs=[5000, 20000, 50000, 100000, 250000, 1000000], max_candidates=128,
                max_revisions=10000, exploration=adaptive_exploration(),
                model=ModelConfig(provider="compatible", model=args.model, base_url=args.base_url),
                device=args.device, budget=research_budget())
            if args.suite9:
                from arlab.tasks.iql.study import apply_nine_task_protocol
                contract = apply_nine_task_protocol(contract.model_dump(mode="json"))
            write_json(args.output, contract)
            _print({"contract": str(args.output), "cases": len(cases), "note": "Review the explicit compute budget before running."})
        elif args.command == "train":
            from arlab.tasks.iql.recipe import IQLRecipe
            from arlab.tasks.protocol import get_task
            from arlab.algorithms.runtime import authorize
            from arlab.algorithms.spec import CodeRecipe, is_code_recipe
            from arlab.core.jobs import LocalBackend, DockerBackend
            import sys
            task = get_task("iql")
            recipe = task.validate_recipe(read_json(args.recipe) if args.recipe else IQLRecipe().model_dump(mode="json"))
            policy = CodeExecutionPolicy(mode=args.code_mode, docker_image=args.code_image,
                                         container_gpu=args.device.startswith("cuda"))
            if is_code_recipe(recipe):
                authorize(CodeRecipe.model_validate(recipe), policy)
            request = {"recipe": recipe, "data": str(args.dataset.resolve()),
                       "dataset_sha256": file_digest(args.dataset), "seed": args.seed, "steps": args.steps,
                       "device": args.device, "threads": 1, "source_sha256": source_digest(),
                       "code_execution": policy.model_dump(mode="json")}
            output = args.output.resolve()
            output.mkdir(parents=True, exist_ok=True)
            request_file = output / "request.json"
            write_json(request_file, request)
            backend = (DockerBackend(policy.docker_image, [request_file, args.dataset.resolve()], output,
                                     code_policy=policy) if policy.mode == "docker" else LocalBackend())
            # Even local CLI candidates run with scrubbed environment in a worker.
            result = backend.run([sys.executable, "-m", "arlab.worker", "train", "--task", "iql",
                                  "--request", str(request_file), "--output", str(output)],
                                 output, output / "worker.log", 86400)
            if result.exit_code:
                raise RuntimeError(f"Training worker failed; inspect {output / 'worker.log'}")
            _print(read_json(output / "training.json"))
        elif args.command == "evaluate":
            from arlab.tasks.iql.evaluator import evaluate_policy
            manifest = read_json(args.policy / "policy.json")
            environment = {"kind": "d4rl", "id": args.env} if args.env != "synthetic_lq" else {
                "kind": "synthetic_lq", "dynamics_seed": args.dynamics_seed, "horizon": args.horizon}
            _print(evaluate_policy({"artifact": str(args.policy.resolve()), "artifact_sha256": directory_digest(args.policy),
                "training_binding": manifest["training_binding"], "steps": manifest["updates"],
                "environment": environment, "eval_seeds": list(range(args.seed, args.seed+args.episodes)),
                "code_execution": CodeExecutionPolicy(mode=args.code_mode, docker_image=args.code_image).model_dump(mode="json")}, args.output))
        elif args.command == "audit":
            from arlab.audit import audit_campaign
            _print(audit_campaign(args.campaign, args.output, split=args.split,
                                 train_seeds=args.train_seeds, eval_seeds=args.eval_seeds,
                                 optimizer_step_limit=args.optimizer_step_limit, dry_run=args.dry_run))
        elif args.command in ("research", "import-source", "context"):
            _research_command(args)
        elif args.command == "export":
            frozen = read_json(args.campaign / "frozen.json")
            args.output.mkdir(parents=True, exist_ok=True)
            write_json(args.output / "recipe.json", frozen["recipe"])
            write_json(args.output / "provenance.json", frozen)
            from arlab.core.io import atomic_write
            if frozen["recipe"].get("kind") == "python":
                from arlab.algorithms.spec import CodeRecipe
                CodeRecipe.model_validate(frozen["recipe"]).materialize(args.output / "source")
                _print({"export": str(args.output.resolve()), "kind": "editable_python_source"})
                return 0
            atomic_write(args.output / "candidate.py", '"""Exported in-sample tensor program; no executable model-generated Python."""\n'
                'import json\nfrom pathlib import Path\nfrom arlab.core.expressions import evaluate\n'
                'RECIPE = json.loads(Path(__file__).with_name("recipe.json").read_text())\n\n'
                'def value_loss(delta, expectile):\n    return evaluate(RECIPE["value_loss"], {"delta": delta, "expectile": expectile})\n\n'
                'def q_loss(td1, td2):\n    return evaluate(RECIPE["q_loss"], {"td1": td1, "td2": td2})\n\n'
                'def actor_weight(adv, beta, weight_cap):\n    return evaluate(RECIPE["actor_weight"], {"adv": adv, "beta": beta, "weight_cap": weight_cap})\n')
            _print({"export": str(args.output.resolve())})
        elif args.command == "doctor":
            import importlib.util
            import platform
            import torch
            result = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.cuda.is_available(),
                      "gym": importlib.util.find_spec("gym") is not None, "d4rl": importlib.util.find_spec("d4rl") is not None,
                      "source_sha256": source_digest()}
            if args.d4rl_env:
                from arlab.deployment.tasks import check_environment
                result["environment_check"] = check_environment(args.d4rl_env)
                result["environment_smoke_passed"] = args.d4rl_env
            _print(result)
        return 0
    except KeyboardInterrupt:
        print("Interrupted. A persisted campaign can be resumed.")
        return 130
    except Exception as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", file=__import__("sys").stderr)
        return 1


def _research_command(args: argparse.Namespace) -> None:
    from arlab.core.harness import Harness
    from arlab.core.schema import Decision
    from arlab.core.store import campaign_lock
    contract = ResearchContract.model_validate(read_json(args.campaign / "contract.json"))
    config = AgentConfig.model_validate(read_json(args.campaign / "agent.json"))
    agent = Harness(contract, args.campaign, config=config)
    try:
        with campaign_lock(args.campaign):
            agent._initialize()
            if args.command == "context":
                _print(agent._context())
            elif args.command == "import-source":
                if args.path.stat().st_size > 500_000 or args.path.is_symlink():
                    raise ValueError("Owner text import requires a regular file under 500 KB")
                text = args.path.read_text(encoding="utf-8")
                sid = agent.research._source(text, {"title": args.title or args.path.name,
                    "url": args.url, "scope": "owner_imported_text", "raw_sha256": file_digest(args.path),
                    "imported_by": "owner", "source_identity_verified": False})
                _print({"source": sid})
            else:
                raw = read_json(args.action)
                allowed = {"search_literature", "search_web", "fetch_source", "read_source", "find_source",
                           "search_repositories", "read_repository"}
                if raw.get("kind") not in allowed:
                    raise ValueError("Only read-only research actions accepted here")
                result = agent._execute(Decision(summary="Owner-invoked research operation", action=raw))
                agent.store.event("owner_research", result)
                _print(result.model_dump(mode="json"))
    finally:
        agent.close()
