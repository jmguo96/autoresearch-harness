"""Student-facing orchestration, using the same Harness and owner-only audit.

One command runs a full study. Simplifying the interface never reduces its final
panel or substitutes synthetic data. Local execution requires explicit informed approval.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import sys
from pathlib import Path

from arlab.classroom.bootstrap import confirm
from arlab.classroom.kimi import (
    HTTP_NOTICE, default_model_settings, prepare_model, validate_key,
)
from arlab.core.endpoints import AUTO_KIMI, is_lab_gateway, validate_model_endpoint
from arlab.classroom.reporting import study_report
from arlab.core.io import digest, file_digest, read_json, source_digest, write_json
from arlab.core.schema import ModelConfig, ResearchContract
from arlab.core.store import campaign_lock
from arlab.tasks.iql.study import require_nine_task_protocol


def validate_endpoint(value: str, *, allow_lab_http: bool = False) -> bool:
    return validate_model_endpoint(value, allow_lab_http=allow_lab_http)


def model_settings(root: Path, args: argparse.Namespace, default: ModelConfig) -> ModelConfig:
    """Persist only non-secret connection settings; never write the API key."""
    path = root / "model.json"
    raw = read_json(path) if path.exists() else default.model_dump(mode="json")
    overrides = any(getattr(args, key, None) for key in ("provider", "model", "base_url"))
    unconfigured = not raw.get("model") or "REPLACE" in raw.get("model", "") or raw.get("provider") == "demo"
    if unconfigured and not overrides:
        raw = default_model_settings()
    for key in ("provider", "model", "base_url"):
        if getattr(args, key, None):
            raw[key] = getattr(args, key)
    # Teacher overrides do not accidentally carry the plaintext grant elsewhere.
    if getattr(args, "base_url", None):
        raw["allow_lab_http"] = is_lab_gateway(args.base_url)

    interactive = sys.stdin.isatty()
    missing = not raw.get("model") or "REPLACE" in raw.get("model", "") or raw.get("provider") == "demo"
    if interactive and missing:
        print("只配置模型服务；论文和公开代码检索无需额外搜索密钥。")
        raw["base_url"] = input(f"模型服务地址 [{raw.get('base_url', '')}]：").strip() or raw.get("base_url", "")
        raw["model"] = input("模型名称：").strip() or raw.get("model", "")
        raw["provider"] = raw.get("provider") if raw.get("provider") in ("compatible", "responses", "anthropic") else "compatible"
    config = ModelConfig.model_validate(raw)
    if config.provider == "demo" or "REPLACE" in config.model or not config.model.strip():
        raise ValueError("请填写真实模型名称；交互运行会提示，自动化使用 --model 和 --base-url。")
    validate_endpoint(config.base_url, allow_lab_http=config.allow_lab_http)
    if config.model == AUTO_KIMI and config.provider != "compatible":
        raise ValueError("自动选择 Kimi 要求 compatible Chat Completions 协议。")
    write_json(path, config)
    path.chmod(0o600)
    return config


def require_key(config: ModelConfig, allow_no_key: bool) -> bool:
    """Return whether this function temporarily put a typed key in the environment."""
    local = validate_endpoint(config.base_url, allow_lab_http=config.allow_lab_http)
    key = os.environ.get(config.api_key_env, "")
    if key:
        validate_key(key)
        return False
    # Previous editions used ARLAB_API_KEY. Copy only inside the owner process;
    # the caller removes this temporary alias on exit, including failure paths.
    if config.api_key_env == "NEW_API_KEY" and os.environ.get("ARLAB_API_KEY"):
        os.environ[config.api_key_env] = validate_key(os.environ["ARLAB_API_KEY"])
        return True
    if local and allow_no_key:
        return False
    if not sys.stdin.isatty():
        raise ValueError(f"请设置 {config.api_key_env}；密钥不会写入配置文件。")
    key = getpass.getpass("API Key（管理员分发，隐藏输入，仅本次进程使用）：").strip()
    if not key:
        raise ValueError("未提供模型凭据。")
    os.environ[config.api_key_env] = validate_key(key)
    return True


# 这是授权上限与容量说明，不是要求模型跑满所有分支的固定调度表。
def budget_preview(contract: ResearchContract) -> dict:
    final = contract.final_evaluation
    if final is None:
        raise ValueError("Missing final panel")
    n = len(final.cases) * len(final.train_seeds)
    upper = contract.code_execution.max_optimizer_steps_per_tick if contract.code_execution.mode != "disabled" else 3
    development_tasks = [c.name for c in contract.cases if c.split == "dev"]
    development_cells = len(development_tasks) * len(contract.exploration.confirmation_seeds or contract.train_seeds)
    return {
        'exploration_tasks': development_tasks,
        'development_cells_per_method_per_rung': development_cells,
        'development_final_rung_runs_for_two_methods': 2 * development_cells,
        'development_all_candidates_all_rungs': {
            'scope': 'NOT a schedule or authorized expenditure. With high branch limits an exhaustive grid is deliberately unaffordable; promote selectively. Checkpoints reuse earlier work.',
            'methods_including_IQL': contract.max_candidates,
            'trials': contract.max_candidates * development_cells * len(contract.rungs),
            'ticks': contract.max_candidates * development_cells * (contract.rungs[-1] if contract.exploration.adaptive else sum(contract.rungs)),
            'eval_episodes': contract.max_candidates * development_cells * len(contract.rungs) * len(contract.eval_seeds),
        },
        'adaptive_start': {
            'tasks': 1,
            'seeds': 1,
            'steps': (
                contract.exploration.initial_steps
                if contract.exploration.adaptive
                else contract.rungs[0]
            ),
            'eval_episodes': (
                contract.exploration.initial_eval_episodes
                if contract.exploration.adaptive
                else len(contract.eval_seeds)
            ),
        },
        'active_branch_limit': contract.max_candidates,
        'revision_limit': contract.max_revisions,
        'schedule_horizon': contract.rungs[-1],
        'selection': 'Model chooses continuation, extra dev tasks/seeds/episodes, revisions or stop. Final confirmation is fixed.',
        'exploration_rungs': contract.rungs,
        'exploration_training_seeds': contract.train_seeds,
        'exploration_budget': contract.budget.model_dump(mode='json'),
        'final_tasks': final.cases,
        'final_training_seeds': final.train_seeds,
        'final_rollout_seeds': final.eval_seeds,
        'final_steps_per_run': final.steps,
        'final_runs_if_candidate_differs': 2 * n,
        'final_runs_if_IQL_retained': n,
        'final_ticks_if_candidate_differs': 2 * n * final.steps,
        'final_optimizer_upper_from_permissions': n * final.steps * (3 + upper),
        'final_optimizer_authorized': final.max_optimizer_steps,
        'final_retry_reserve_fraction': final.retry_reserve_fraction,
        'final_max_attempts_per_trial': final.max_attempts_per_trial,
        'code_execution': contract.code_execution.mode,
        'literature_requires_extra_key': False,
        'general_web_search': False,
        'warning': 'This is a complete research budget, NOT smoke training. Full coverage does not guarantee improvement.',
    }


def _deployment_contract(root: Path) -> ResearchContract:
    state = read_json(root / "deployment.json")
    checks = read_json(root / "checks.json")
    if state.get("status") != "ready" or checks.get("source_sha256") != source_digest():
        raise ValueError("环境自检未完成或实现版本变化；使用新目录运行 setup。")
    contract = ResearchContract.model_validate(read_json(root / "configs/research.json"))
    require_nine_task_protocol(contract)
    for case in contract.cases:
        if not case.dataset_sha256 or file_digest(Path(case.data)) != case.dataset_sha256:
            raise ValueError(f"数据缺失或改变：{case.name}，未调用模型或启动实验。")
    return contract


# 学生主线只有两段：动态开发研究 → 冻结后的程序化最终确认。
# 最终成绩不送回已结束的研究者，缺任何阶段的证据都不能报告完整研究。
def run_study(root: Path, contract: ResearchContract) -> dict:
    """Synchronous full study; exceptions leave a partial report, never success."""
    from arlab.audit import audit_campaign
    from arlab.core.harness import Harness
    with campaign_lock(root):
        try:
            path = root / "development/report.json"
            development = read_json(path) if path.exists() else {}
            if development.get("status") != "completed":
                count = sum(c.split == "dev" for c in contract.cases)
                print(f"\n[研究] 文献/代码调研、{count} 任务候选探索、匹配 IQL 对照。", flush=True)
                agent = Harness(contract, root / "development")
                from arlab.classroom.output import ProgressModel
                agent.model = ProgressModel(agent.model)
                try:
                    development = agent.run()
                finally:
                    agent.close()
            # 开发未完成就保留部分报告；不能用单任务或局部端点冒充最终结论。
            if development.get("status") != "completed":
                return study_report(root, contract)
            print("\n[确认] 方法已冻结；重新训练 IQL 和候选，执行完整最终面板。", flush=True)
            preview = audit_campaign(root / "development", root / "final", split="final", dry_run=True)
            print(json.dumps(preview["required"], ensure_ascii=False, indent=2), flush=True)
            if not preview["authorized"]:
                raise ValueError("最终验证资源超出预授权；没有启动最终训练。")
            audit_campaign(root / "development", root / "final", split="final")
            return study_report(root, contract)
        except BaseException:
            study_report(root, contract)
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "configure", "report", "finalize", "demo"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--name", default="study")
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--provider", choices=("compatible", "responses", "anthropic"))
    parser.add_argument("--confirm-budget", action="store_true")
    parser.add_argument("--allow-no-api-key", action="store_true")
    parser.add_argument("--offline-research", action="store_true")
    parser.add_argument("--acknowledge-local-code", action="store_true")
    args = parser.parse_args(argv)
    typed_key, key_name = False, ""
    try:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", args.name):
            raise ValueError("实验名只允许字母、数字、下划线、短横线。")
        if args.action == "demo":
            result = read_json(args.root / "smoke-latest.json")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print("这是合成环境和确定性决策器；不生成九任务研究结论。")
            return 0
        deployment = _deployment_contract(args.root)
        study = args.root / "runs" / args.name
        existing = study / "development/contract.json"
        if args.action == "configure":
            model_settings(args.root, args, deployment.model)
            print("非秘密模型设置已保存；Kimi 自动选择将在填写 Key 后验证。运行 bash start.sh 开始研究。")
            return 0
        contract = ResearchContract.model_validate(read_json(existing)) if existing.exists() else deployment
        require_nine_task_protocol(contract)
        if args.action == "report":
            if not existing.exists():
                raise ValueError("研究尚未启动；没有可展示的研究结论。")
            print(json.dumps(study_report(study, contract), ensure_ascii=False, indent=2))
            print(f"报告：{study / 'report.md'}")
            return 0
        if (study / "final/audit.json").exists():
            result = study_report(study, contract)
            print(f"已有最终结果：{study / 'report.md'}；不会再次训练或请求模型。")
            return 0 if result["research_complete"] else 2
        dev_report = read_json(study / "development/report.json") if (study / "development/report.json").exists() else {}
        needs_model = dev_report.get("status") != "completed"
        if args.action == "finalize" and needs_model:
            raise ValueError("尚未冻结完整匹配的开发候选，请先运行默认 start.sh。")
        if existing.exists():
            if any(getattr(args, k) and getattr(args, k) != getattr(contract.model, k) for k in ("model", "base_url", "provider")):
                raise ValueError("不能在已有研究中更换模型；新研究使用 --name 新名称。")
            if args.offline_research and contract.research_access.enabled:
                raise ValueError("已有研究合同不可更改调研权限；请用新实验名称。")
        elif needs_model:
            raw = deployment.model_dump(mode="json")
            raw["model"] = model_settings(args.root, args, deployment.model).model_dump(mode="json")
            raw["research_access"]["web_search_enabled"] = False
            if args.offline_research:
                raw["research_access"]["enabled"] = False
            contract = ResearchContract.model_validate(raw)
        print(json.dumps(budget_preview(contract), ensure_ascii=False, indent=2))
        managed_local = bool(os.environ.get("ARLAB_LOCAL_ROOT"))
        if contract.code_execution.mode == "docker" and managed_local:
            raise ValueError("本安装版不使用 Docker；请创建新的本地 uv 部署。")
        if contract.code_execution.mode == "trusted_local" and managed_local:
            from arlab.deployment.local import LOCAL_CODE_NOTICE
            print(LOCAL_CODE_NOTICE, flush=True)
            confirm("确认上述预算，以及在项目虚拟环境中本地执行候选 Python 的风险？", granted=args.confirm_budget)
        else:
            confirm("允许使用上述探索预算，并在方法冻结后执行完整最终评价？", granted=args.confirm_budget)
            if contract.code_execution.mode == "trusted_local" and not args.acknowledge_local_code:
                raise ValueError("本地代码不是沙箱；已审阅代码才可加 --acknowledge-local-code。")
        if needs_model:
            if is_lab_gateway(contract.model.base_url):
                print(HTTP_NOTICE, flush=True)
            key_name = contract.model.api_key_env
            typed_key = require_key(contract.model, args.allow_no_api_key)
            resolved = prepare_model(contract.model)
            if existing.exists() and resolved != contract.model:
                raise ValueError("已有研究的模型不能重新选择；请保留原模型。")
            if resolved != contract.model:
                raw = contract.model_dump(mode="json")
                raw["model"] = resolved.model_dump(mode="json")
                contract = ResearchContract.model_validate(raw)
            # Save only verified settings. The key is never part of ModelConfig.
            write_json(args.root / "model.json", resolved)
            (args.root / "model.json").chmod(0o600)
            print(f"研究模型已固定：{resolved.model}（{resolved.provider}）", flush=True)
        result = run_study(study, contract)
        print(f"\n状态：{result['status']}\n报告：{study / 'report.md'}\n完整数据：{study / 'report.json'}")
        return 0 if result["research_complete"] else 2
    except KeyboardInterrupt:
        print("\n已暂停，重复同一命令继续。当前证据不能替代最终九项结论。", file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"研究提示：{exc}", file=sys.stderr)
        return 2
    finally:
        if typed_key:
            os.environ.pop(key_name, None)
