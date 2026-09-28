"""Owner-only confirmation. Freeze one method, then evaluate it and fixed IQL.

The researcher has no audit tool. A final panel may contain its development task;
this overlap is disclosed, never called an entirely unseen benchmark.
"""
from __future__ import annotations

import time
import math
from pathlib import Path
from typing import Any

from arlab.core.experiments import TrialService
from arlab.core.io import digest, read_json, source_digest, write_json
from arlab.core.schema import Budget, ResearchContract, RunAction
from arlab.core.stats import paired_comparison, summarize
from arlab.core.store import Store, campaign_lock, BudgetExceeded
from arlab.tasks.protocol import get_task


# 确认 1｜固定方法与任务/seed/端点组成审计协议；final 禁止调用时覆盖条件。
def _protocol(original: ResearchContract, frozen: dict, output: Path, split: str,
              train_seeds: list[int] | None, eval_seeds: list[int] | None,
              optimizer_step_limit: int | None) -> tuple[list, dict, dict]:
    final = original.final_evaluation if split == "final" else None
    if split == "final":
        if final is None:
            raise ValueError("No predeclared final evaluation panel")
        if train_seeds is not None or eval_seeds is not None or optimizer_step_limit is not None:
            raise ValueError("Final protocol is frozen before search; per-call overrides are forbidden")
        by_name = {c.name: c for c in original.cases}
        cases = [by_name[name] for name in final.cases]
        seeds, episodes, steps = final.train_seeds, final.eval_seeds, final.steps
        limit = final.max_optimizer_steps
        wall, timeout = final.max_wall_seconds, final.worker_timeout_seconds
        retries = final.max_attempts_per_trial
        retry_fraction = final.retry_reserve_fraction
    else:
        cases = [c for c in original.cases if c.split == split]
        seeds = [41, 43, 47, 53, 59] if train_seeds is None else train_seeds
        episodes = list(range(9000, 9010)) if eval_seeds is None else eval_seeds
        steps = original.rungs[-1]
        limit = original.budget.max_optimizer_steps if optimizer_step_limit is None else optimizer_step_limit
        wall = original.budget.max_wall_seconds
        timeout = max(original.budget.worker_timeout_seconds, 3600)
        retries = original.budget.max_attempts_per_trial
        retry_fraction = 0.25
    if not cases or not seeds or not episodes:
        raise ValueError("Audit requires a nonempty task/seed/episode panel")
    if len(set(seeds)) != len(seeds) or len(set(episodes)) != len(episodes):
        raise ValueError("Audit seeds must be unique")
    # 相同方法按内容去重；保留 IQL 时不会把同一方法训练两遍。
    unique = {
        digest(original.seed_recipe): original.seed_recipe,
        digest(frozen['recipe']): frozen['recipe'],
    }
    cells = len(unique) * len(cases) * len(seeds)
    optimizer_required = len(cases) * len(seeds) * steps * sum(
        int(r.get("max_optimizer_steps_per_tick", 3)) for r in unique.values())
    estimate = {
        'trials': cells,
        'train_steps': cells * steps,
        'optimizer_steps': optimizer_required,
        'optimizer_step_limit': limit,
        'eval_episodes': cells * len(episodes),
    }
    # Recovery headroom is pre-authorized, not free compute or a silent refund.
    # A pool (25% by default, at least one cell's allowed retries) can be spent on
    # any failed cell, subject also to the persistent per-cell attempt cap.
    retry_slots = max(retries - 1, math.ceil(cells * retry_fraction)) if retries > 1 else 0
    maximum_bound = max(int(r.get("max_optimizer_steps_per_tick", 3)) for r in unique.values())
    # 名义面板与有限恢复容量分开，恢复池也必须受用户硬上限约束。
    capacity = {
        'attempts': cells + retry_slots,
        'trials': cells,
        'train_steps': estimate['train_steps'] + retry_slots * steps,
        'optimizer_steps': optimizer_required + retry_slots * steps * maximum_bound,
        'eval_episodes': estimate['eval_episodes'] + retry_slots * len(episodes),
    }
    estimate["recovery_capacity"] = capacity
    estimate["retry_slots"] = retry_slots
    estimate["max_attempts_per_trial"] = retries
    protocol = {
        'split': split,
        'cases': [c.name for c in cases],
        'train_seeds': seeds,
        'eval_seeds': episodes,
        'steps': steps,
        'frozen': digest(frozen),
        'output': str(output),
        'budget_estimate': estimate,
        'max_wall_seconds': wall,
        'worker_timeout_seconds': timeout,
        'development_tasks_in_final': [c.name for c in cases if c.split == 'dev'],
    }
    return cases, protocol, estimate


def audit_campaign(campaign: Path, output: Path, *, split: str = "test",
                   train_seeds: list[int] | None = None, eval_seeds: list[int] | None = None,
                   optimizer_step_limit: int | None = None, dry_run: bool = False) -> dict[str, Any]:
    """One immutable protocol per (campaign, split), resumable in the same workspace.

    `final` follows the contract's complete final_evaluation, including development
    tasks with new seeds. It does not select the best method using test scores.
    """
    if split not in ("validation", "test", "final"):
        raise ValueError("Audit split must be validation, test or predeclared final")
    campaign, output = campaign.resolve(), output.resolve()
    if output == campaign or output.is_relative_to(campaign) or campaign.is_relative_to(output):
        raise ValueError("Use a separate, non-overlapping auditor workspace")
    original = ResearchContract.model_validate(read_json(campaign / "contract.json"))
    frozen = read_json(campaign / "frozen.json")
    if frozen["contract_sha256"] != digest(original) or frozen["source_sha256"] != source_digest():
        raise ValueError("Frozen method, contract, or implementation changed")
    cases, protocol, estimate = _protocol(original, frozen, output, split, train_seeds,
                                          eval_seeds, optimizer_step_limit)
    seeds, episodes, steps = protocol["train_seeds"], protocol["eval_seeds"], protocol["steps"]
    if dry_run:
        return {
            'status': 'preview',
            'required': estimate,
            'protocol': protocol,
            'authorized': estimate['recovery_capacity']['optimizer_steps'] <= estimate['optimizer_step_limit'],
        }
    if estimate["recovery_capacity"]["optimizer_steps"] > estimate["optimizer_step_limit"]:
        raise ValueError("Audit requires more optimizer steps than authorized; refused before starting any work")
    # The child execution contract has different seeds, but cannot redefine the
    # study contract. Keep its final protocol in `protocol`, not on the child.
    raw = original.model_dump(mode="json")
    raw.update(
        train_seeds=seeds,
        eval_seeds=episodes,
        rungs=[steps],
        final_evaluation=None,
        exploration={
            'adaptive': False,
        },
    )
    capacity = estimate["recovery_capacity"]
    raw["budget"] = Budget(max_trials=capacity['trials'], max_attempts=capacity['attempts'], max_attempts_per_trial=estimate['max_attempts_per_trial'], max_train_steps=capacity['train_steps'], max_optimizer_steps=max(1, capacity['optimizer_steps']), max_eval_episodes=capacity['eval_episodes'], max_wall_seconds=protocol['max_wall_seconds'], worker_timeout_seconds=protocol['worker_timeout_seconds']).model_dump(
        mode='json',
    )
    contract = ResearchContract.model_validate(raw)  # Before any external execution.
    output.mkdir(parents=True, exist_ok=True)
    with campaign_lock(campaign), campaign_lock(output):
        parent = Store(campaign / "state.sqlite3")
        try:
            if split == "final" and (parent.get("status") != "completed" or not parent.get("frozen")):
                raise ValueError("Final evaluation requires a completed baseline-matched development selection")
            parent.bind("audit_" + split, protocol)
            parent.expose("frozen_selection", "independent_auditor", split)
            data_hashes = parent.get("dataset_hashes")
        finally:
            parent.close()
        if (output / "audit.json").exists():
            report = read_json(output / "audit.json")
            if report["protocol_sha256"] != digest(protocol):
                raise ValueError("Audit output protocol changed")
            return report
        task = get_task(contract.task)
        store = Store(output / "audit.sqlite3")
        progress = {
            'status': 'running',
            'protocol_sha256': digest(protocol),
            'expected_executions': estimate['trials'],
            'complete_executions': 0,
            'final_conclusion_available': False,
        }
        try:
            store.bind("source_sha256", frozen["source_sha256"])
            store.bind("dataset_hashes", data_hashes)
            store.bind("protocol", protocol)
            store.put("started_at", store.get("started_at", time.time()))
            baseline = store.add_candidate(task.validate_recipe(original.seed_recipe), {"role": "baseline"})
            selected = store.add_candidate(
                task.validate_recipe(frozen['recipe']),
                {
                    'role': 'frozen_selection',
                },
                baseline,
            )
            service = TrialService(output, contract, store, task)
            all_results = []
            for cid in dict.fromkeys([baseline, selected]):
                for case in cases:
                    for seed in seeds:
                        progress.update(current_case=case.name, current_seed=seed, current_candidate=cid)
                        write_json(output / "progress.json", progress)
                        print(
                            f"[最终评价 {len(all_results) + 1}/{estimate['trials']}] {case.name} seed={seed} candidate={cid}",
                            flush=True,
                        )
                        for retry in range(contract.budget.max_attempts_per_trial):
                            try:
                                result = service.run(
                                    RunAction(
                                        kind='run',
                                        candidate=cid,
                                        case=case.name,
                                        train_seed=seed,
                                        steps=steps,
                                    ),
                                    authorized_split=case.split,
                                    observer='independent_auditor',
                                )
                                break
                            except BudgetExceeded:
                                raise
                            except (RuntimeError, OSError) as exc:
                                # Value/constraint errors are not retried as transient outages.
                                progress.update(retry=retry + 1, last_error_type=type(exc).__name__)
                                store.event(
                                    'audit_retry',
                                    {
                                        'case': case.name,
                                        'candidate': cid,
                                        'seed': seed,
                                        'retry': retry + 1,
                                        'error': str(exc)[-1000:],
                                    },
                                )
                                write_json(output / "progress.json", progress)
                                if retry + 1 >= contract.budget.max_attempts_per_trial:
                                    raise
                        all_results.append(result)
                        progress["complete_executions"] = len(all_results)
                        write_json(output / "progress.json", progress)
            b = [t for t in all_results if t["candidate"] == baseline]
            c = [t for t in all_results if t["candidate"] == selected]
            expected = {(case.name, seed, steps) for case in cases for seed in seeds}
            comparison = paired_comparison(b, c, expected_cells=expected)
            if not comparison["contract_panel_complete"]:
                raise ValueError("Incomplete confirmation panel: final report was not published")
            seen = set(protocol["development_tasks_in_final"])
            unseen = {case.name for case in cases} - seen
            development_comparison = paired_comparison(
                [t for t in b if t['case'] in seen],
                [t for t in c if t['case'] in seen],
                expected_cells={x for x in expected if x[0] in seen},
            )
            unseen_comparison = paired_comparison(
                [t for t in b if t['case'] in unseen],
                [t for t in c if t['case'] in unseen],
                expected_cells={x for x in expected if x[0] in unseen},
            )
            report = {
                'status': 'completed',
                'protocol_sha256': digest(protocol),
                'protocol': protocol,
                'baseline': summarize(b),
                'selected': summarize(c),
                'comparison': comparison,
                'development_task_comparison': development_comparison,
                'unseen_task_comparison': unseen_comparison,
                'selected_candidate': selected,
                'baseline_candidate': baseline,
                'trials': all_results,
                'development_tasks_in_final': sorted(seen),
                'unseen_tasks': sorted(unseen),
                'reserved': store.totals(),
                'actual_known_usage': store.totals(True),
                'final_conclusion_available': split == 'final',
                'evaluator_calls_llm': False,
                'warning': 'Completed does not imply improvement. Developed tasks are not held-out; unseen_tasks only means unused by this campaign search. If any of these scores guide another version, this panel becomes development evidence for that version; renaming a run or changing seeds does not restore unseen tasks.',
            }
            write_json(output / "audit.json", report)
            progress.update(status="completed", final_conclusion_available=split == "final")
            write_json(output / "progress.json", progress)
            return report
        except BaseException as exc:
            progress.update(
                status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'incomplete',
                error_type=type(exc).__name__,
                final_conclusion_available=False,
            )
            write_json(output / "progress.json", progress)
            raise
        finally:
            store.close()
