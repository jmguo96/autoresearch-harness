"""A dependency-free classroom entry, sharing the existing owner deployment path.

The default is real D4RL, not synthetic data. Installation approval and model /
research spending approval remain separate. No commands are built with shell=True.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from arlab.classroom.constants import D4RL_SUITE9, EXPLORATION_TASKS
from arlab.deployment.common import public_environment, read_json
from arlab.deployment.local import LOCAL_CODE_NOTICE, local_process


def parser(*, show_advanced: bool = False) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bash start.sh",
        description="默认：准备 D4RL → 填写实验室 API Key → 确认模型 → 确认预算并研究。",
        epilog="教师可先运行 setup；学生通常只需再次运行 bash start.sh。详细选项见 docs/quickstart.md。",
    )
    p.add_argument("action", nargs="?", default="run", choices=("run", "setup", "demo", "report", "configure", "finalize", "check-api"),
                   help="run 默认研究；setup 只安装；demo 无密钥自检；report 看结果；configure 教师配置；check-api 检查当前模型可见性")
    p.add_argument("--advanced-help", action="store_true", help="查看教师/自动化选项")
    p.add_argument("--dry-run", action="store_true", help="只预览，不联网、不安装、不写文件")
    p.add_argument("--root", type=Path, help="项目内的部署目录，不允许指向其他项目")
    # Advanced options are intentionally not part of the first-screen walkthrough.
    advanced = p.add_argument_group("教师/自动化选项")
    def teacher_argument(*flags, **kwargs):
        if not show_advanced:
            kwargs["help"] = argparse.SUPPRESS
        return advanced.add_argument(*flags, **kwargs)
    teacher_argument("--yes", action="store_true", help="确认安装，不代表确认模型/研究预算")
    teacher_argument("--confirm-budget", action="store_true")
    teacher_argument("--model")
    teacher_argument("--base-url")
    teacher_argument("--provider", choices=("compatible", "responses", "anthropic"))
    teacher_argument("--offline-research", action="store_true", help="明确关闭外部资料请求，不影响已保存资料读取")
    teacher_argument("--native-deps", choices=("auto", "check"), default="auto", help="缺少原生库时自动项目内下载解包；check仅检查")
    teacher_argument("--native-offline", action="store_true", help="只复用已经锁定下载的原生deb")
    teacher_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    teacher_argument("--code-mode", choices=("disabled", "trusted_local"), default="trusted_local")
    teacher_argument("--acknowledge-local-code", action="store_true")
    teacher_argument("--allow-no-api-key", action="store_true", help="只允许回环地址上的无鉴权模型")
    teacher_argument("--name", default="study", help="新实验用新名称；同名继续已有状态")
    teacher_argument("--reuse-runtime", type=Path, help=argparse.SUPPRESS)
    return p


def confirm(message: str, *, granted: bool = False) -> None:
    if granted:
        return
    if not sys.stdin.isatty():
        raise ValueError(message + "；非交互终端需要显式授权参数。")
    if input(message + " [y/N] ").strip().lower() not in ("y", "yes"):
        raise ValueError("已取消，没有执行未授权操作。")


def installation_command(source: Path, root: Path, args: argparse.Namespace) -> list[str]:
    demo = args.action == "demo"
    command = ["bash", str(source / "deploy.sh"), "--yes", "--root", str(root),
               "--profile", "smoke" if demo else "d4rl", "--device", args.device,
               "--smoke-steps", "12" if demo else "40"]
    if not demo:
        command += ["--suite9", "--code-mode", args.code_mode, "--native-deps", args.native_deps]
        if args.native_offline:
            command.append("--native-offline")
        if not args.offline_research:
            command.append("--research-online")
        if args.code_mode == "trusted_local":
            command.append("--acknowledge-local-code")
    if args.reuse_runtime:
        if not demo:
            raise ValueError("--reuse-runtime 仅用于明确标记的合成自检，不是 D4RL 安装。")
        command += ["--reuse-runtime", str(args.reuse_runtime)]
    return command


def require_current_source(root: Path, source: Path) -> dict:
    from arlab.deployment.manager import _source_identity
    state = read_json(root / "deployment.json")
    if state.get("status") != "ready":
        raise ValueError("部署没有通过真实训练自检，请重新执行 bash start.sh setup。")
    if state.get("source_identity") != _source_identity(source):
        raise ValueError("源码版本与部署不同；请用 --root 指定新目录安装，旧实验保留不变。")
    return state


def main(argv: list[str] | None = None, *, source: Path) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--advanced-help" in arguments:
        parser(show_advanced=True).print_help()
        return 0
    args = parser().parse_args(arguments)
    try:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", args.name):
            raise ValueError("实验名称只能包含字母、数字、下划线和短横线。")
        if args.action == "check-api":
            from arlab.classroom.kimi import check_api
            from arlab.core.endpoints import AUTO_KIMI, LAB_NEW_API_BASE_URL
            configured = {}
            selected_root = (args.root or source / ".arlab" / "d4rl9").expanduser().absolute()
            for path in (selected_root / "runs" / args.name / "development" / "contract.json",
                         selected_root / "model.json"):
                if path.is_file():
                    value = read_json(path)
                    configured = (value.get("model", {}) if "cases" in value else value) if isinstance(value, dict) else {}
                    break
            base_url = args.base_url or configured.get("base_url") or LAB_NEW_API_BASE_URL
            model = args.model or configured.get("model") or AUTO_KIMI
            provider = args.provider or configured.get("provider") or "compatible"
            if args.dry_run:
                print(json.dumps({"endpoint": base_url.rstrip("/") + "/models",
                    "model_selector": model, "provider": provider, "writes_or_installs": False,
                    "calls_model_now": False, "network_request_now": False}, ensure_ascii=False, indent=2))
                return 0
            return check_api(base_url=base_url, model=model, provider=provider)
        profile = "smoke" if args.action == "demo" else "d4rl"
        root = (args.root or source / ".arlab" / ("smoke" if profile == "smoke" else "d4rl9")).expanduser().absolute()
        # Reuse the same path restrictions as the real deployment even for previews.
        from arlab.deployment.manager import parser as deploy_parser, validate_options
        command = installation_command(source, root, args)
        validate_options(deploy_parser().parse_args(command[2:]), source)
        if args.dry_run:
            print(json.dumps({"action": args.action, "deployment": str(root),
                "installation_command_if_needed": command, "research_preset": "full_nine_task_confirmation",
                "literature_and_public_code": not args.offline_research and args.action != "demo",
                "extra_search_key_required": False, "general_web_search": False,
                "exploration_tasks": ["synthetic_fixture_NOT_D4RL"] if args.action == "demo" else list(EXPLORATION_TASKS),
                "development_cells_per_method_per_rung": None if args.action == "demo" else 9,
                "final_tasks": [] if args.action == "demo" else list(D4RL_SUITE9),
                "final_training_seeds": 5, "final_steps_per_run": 1_000_000,
                "final_training_jobs": 90, "note_final": "IQL+one different frozen candidate; choosing IQL itself requires 45 jobs, not duplicated runs",
                "calls_model_now": False, "writes_or_installs": False,
                "uses_sudo": False, "uses_docker": False, "installer": "uv",
                "installation_scope": "project_local", "execution_notice": LOCAL_CODE_NOTICE,
                "default_model_family": "Kimi", "model_id_policy": "resolve from /v1/models, then pin",
                "api_key_env": "NEW_API_KEY", "default_base_url": "http://43.143.241.66:18319/v1",
                "plaintext_http_notice": "Gateway HTTP is not encrypted; use an administrator-protected network or HTTPS gateway.",
                "note": "学生只填实验室 API Key；地址已预置，真实 Kimi ID 从模型列表确认。安装不等于研究。"},
                ensure_ascii=False, indent=2))
            return 0
        state_path = root / "deployment.json"
        state = read_json(state_path) if state_path.exists() else {}
        needs_install = state.get("status") != "ready" or args.action == "setup"
        if needs_install and args.action in ("report", "configure", "finalize", "check-api"):
            raise ValueError("尚未准备环境，请先运行 bash start.sh setup。")
        if needs_install:
            print("\n" + ("无密钥自检：合成环境＋教学决策器，训练与评价真实执行。" if args.action == "demo"
                  else "D4RL：3 任务探索，覆盖 3 个环境及 medium / medium-replay / medium-expert，最终完整 9 项 × 5 训练种子；调研免额外密钥；关闭通用网页搜索。"), flush=True)
            print(LOCAL_CODE_NOTICE, flush=True)
            confirm("允许用 uv 在本项目安装、下载数据并本地运行小规模自检？", granted=args.yes)
            # The deployer scopes HOME/caches itself after it validates root ownership.
            result = subprocess.run(command, env=public_environment(), check=False)
            if result.returncode:
                print(f"部署未完成；日志保存在 {root / 'attempts'}，修复后重复同一命令。", file=sys.stderr)
                return result.returncode
        state = require_current_source(root, source)
        if args.action == "setup":
            print("\n环境准备完成；运行 bash start.sh，只需填写实验室 API Key。Kimi 地址已预置；安装没有调用大模型。")
            return 0
        if args.action in ("run", "configure", "finalize", "check-api") and state.get("profile") != "d4rl":
            raise ValueError("当前目录是合成自检，不会伪装为 D4RL；使用默认目录或指定 D4RL --root。")
        wrapper = root / "bin/arlab"
        if not wrapper.is_file():
            raise ValueError("部署缺少运行入口，请重新运行 setup。")
        session = [str(wrapper), "classroom", args.action, "--root", str(root), "--name", args.name]
        for key in ("model", "base_url", "provider"):
            value = getattr(args, key)
            if value:
                session += ["--" + key.replace("_", "-"), value]
        for key in ("confirm_budget", "offline_research", "allow_no_api_key", "acknowledge_local_code"):
            if getattr(args, key):
                session.append("--" + key.replace("_", "-"))
        # Preserve keys only for the owner process, never for installation or workers.
        with local_process(root, source):
            return subprocess.run(session, check=False).returncode
    except KeyboardInterrupt:
        print("\n已中断；已提交的实验状态保留，重复同一命令继续。", file=sys.stderr)
        return 130
    except (ValueError, OSError, KeyError) as exc:
        print(f"启动提示：{exc}", file=sys.stderr)
        return 2
