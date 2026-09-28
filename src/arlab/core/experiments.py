"""Plan-to-experiment boundary: validate, reserve, execute, independently evaluate, commit."""
from __future__ import annotations

import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from arlab.core.io import digest, directory_digest, file_digest, read_json, write_json
from arlab.core.jobs import DockerBackend, LocalBackend
from arlab.core.schema import ResearchContract, RunAction, TrialResult
from arlab.core.checkpoints import training_binding, series_binding
from arlab.core.store import BudgetExceeded
from arlab.core.store import Store
from arlab.tasks.protocol import TaskPack


def limits(contract: ResearchContract) -> dict[str, float]:
    b = contract.budget
    return {
        'attempts': b.max_attempts,
        'checks': b.max_checks,
        'research_requests': contract.research_access.max_requests,
        'research_bytes': contract.research_access.max_download_bytes,
        'trials': b.max_trials,
        'train_steps': b.max_train_steps,
        'optimizer_steps': b.max_optimizer_steps,
        'eval_episodes': b.max_eval_episodes,
        'model_calls': b.max_model_calls,
        'output_tokens': b.max_output_tokens,
    }


class TrialService:
    def __init__(self, root: Path, contract: ResearchContract, store: Store, task: TaskPack,
                 *, docker_image: str | None = None):
        self.root, self.contract, self.store, self.task = root, contract, store, task
        self.docker_image = docker_image

    # 进程边界：训练和独立评价分开；候选策略推理受其代码执行策略限制。
    def _worker(self, mode: str, request: Path, output: Path, read_only: list[Path]) -> float:
        output.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            '-m',
            'arlab.worker',
            mode,
            '--task',
            self.task.name,
            '--request',
            str(request),
            '--output',
            str(output),
        ]
        code_policy = self.contract.code_execution
        if mode == "evaluate" and code_policy.mode != "disabled":
            # The independent evaluator launches only candidate inference in its own
            # restricted container. Do not mount a Docker socket into an evaluator.
            backend = LocalBackend()
        elif mode == "train" and code_policy.mode == "docker":
            backend = DockerBackend(
                code_policy.docker_image,
                [request, *read_only],
                output,
                code_policy=code_policy,
            )
        else:
            backend = DockerBackend(self.docker_image, [request, *read_only], output) if self.docker_image else LocalBackend()
        duration = self.contract.budget.worker_timeout_seconds
        started = self.store.get("started_at", time.time())
        remaining = self.contract.budget.max_wall_seconds - (time.time() - started)
        if remaining <= 0:
            from arlab.core.store import BudgetExceeded
            raise BudgetExceeded("Campaign wall-clock limit reached")
        result = backend.run(command, output, output / "worker.log", min(duration, remaining))
        if result.exit_code != 0:
            error = read_json(output / "error.json") if (output / "error.json").exists() else {}
            message = error.get("message", "worker timed out" if result.timed_out else "worker failed")
            trace = error.get("traceback", "")
            if trace:
                message = f"{error.get('type', 'RuntimeError')}: {message}\n{trace[-3500:]}"
            error_type = error.get("type", "RuntimeError")
            if error_type in ("ValueError", "FloatingPointError"):
                raise ValueError(message)
            raise RuntimeError(message)
        return result.seconds

    def _training_receipt(self, row: dict, series: str) -> tuple[Path, dict] | None:
        """Host checks completed immutable snapshots, never deserializes a checkpoint."""
        if row.get("purpose", "experiment") != "experiment" or row["status"] != "complete":
            return None
        path = self.root / "jobs" / row["trial_id"] / "train"
        if not (path / "training.json").is_file():
            return None
        trained = read_json(path / "training.json")
        if trained.get("series_sha256") != series:
            return None
        if (trained["request_sha256"] != row["request_sha256"] or
                directory_digest(path / "artifact") != trained["artifact_sha256"] or
                (path / "checkpoint.pt").is_symlink() or
                file_digest(path / "checkpoint.pt") != trained.get("checkpoint_sha256")):
            raise ValueError("Reusable training checkpoint/artifact was altered")
        return path, trained

    # steps 是累计目标端点。继续同一轨迹也走 run 的授权、身份和计费路径。
    def continue_trial(self, action) -> dict[str, Any]:
        if action.trial in set((self.store.get("source_migration") or {}).get("legacy_trials", [])):
            raise ValueError("Historical trial remains evidence but its old source checkpoint cannot continue; use run for a new trajectory")
        source = self.store.trial(action.trial)
        if not source or source["status"] != "complete" or source.get("purpose") == "check":
            raise ValueError("Continue requires a completed scored development trial, not a check")
        if not self.contract.exploration.adaptive:
            raise PermissionError("Continuation is not enabled for this frozen contract")
        if action.steps < source["steps"]:
            raise ValueError("Continuation target is cumulative and cannot move backwards")
        return self.run(
            RunAction(
                kind='run',
                candidate=source['candidate'],
                case=source['case'],
                train_seed=source['train_seed'],
                steps=action.steps,
                eval_episodes=action.eval_episodes,
            ),
            continuation=source,
        )

    def run(self, action: RunAction, *, authorized_split: str = "dev", check: bool = False,
            observer: str | None = None, continuation: dict | None = None) -> dict[str, Any]:
        # 实验 1｜检查候选、任务分区、种子、训练端点和评价次数。
        # 局部探索可以少做实验，但不能越过合同的数据与权限范围。
        contract = self.contract
        candidate = self.store.candidate(action.candidate)
        if candidate is None:
            raise ValueError("Unknown candidate")
        matches = [c for c in contract.cases if c.name == action.case]
        if len(matches) != 1:
            raise ValueError("Unknown case")
        case = matches[0]
        if case.split != authorized_split:
            raise PermissionError("Researcher cannot run or observe non-development cases")
        adaptive = contract.exploration.adaptive and not check
        legal_steps = (contract.exploration.minimum_steps <= action.steps <= contract.rungs[-1]
                       if adaptive else action.steps in contract.rungs)
        if action.train_seed not in contract.train_seeds or (not check and not legal_steps):
            raise PermissionError("Seed or training horizon is outside the frozen contract")
        if check and (authorized_split != "dev" or action.steps > contract.budget.max_check_ticks):
            raise PermissionError("Candidate check exceeds the owner-granted check limit")
        default_episodes = (contract.exploration.initial_eval_episodes
                            if adaptive and action.steps < contract.rungs[-1] else len(contract.eval_seeds))
        episodes = 1 if check else (action.eval_episodes or default_episodes)
        if not 1 <= episodes <= len(contract.eval_seeds):
            raise PermissionError("Evaluation episodes exceed the authorized rollout seed pool")
        if not adaptive and not check and episodes != len(contract.eval_seeds):
            raise PermissionError("This non-adaptive protocol requires its complete evaluation episodes")
        eval_seeds = contract.eval_seeds[:episodes]
        purpose = "check" if check else "experiment"
        # 实验 2｜验证算法和数据，建立本端点身份及训练轨迹身份。
        # 自适应轨迹的 schedule_steps 始终固定，不随本次暂停位置缩短。
        recipe = self.task.validate_recipe(candidate["recipe"])
        code_candidate = recipe.get("kind") == "python"
        if code_candidate:
            from arlab.algorithms.runtime import authorize
            from arlab.algorithms.spec import CodeRecipe
            authorize(CodeRecipe.model_validate(recipe), contract.code_execution)
        optimizer_bound = recipe["max_optimizer_steps_per_tick"] if code_candidate else 3
        expected_data_hash = case.dataset_sha256 or self.store.get("dataset_hashes", {}).get(case.name, "")
        if not expected_data_hash or file_digest(Path(case.data)) != expected_data_hash:
            raise ValueError("Missing or changed dataset binding")
        self.task.audit_case(case)
        train_request = {
            'recipe': recipe,
            'data': case.data,
            'dataset_sha256': expected_data_hash,
            'seed': action.train_seed,
            'steps': action.steps,
            'device': contract.device,
            'threads': 1,
            'source_sha256': self.store.get('source_sha256'),
            'code_execution': contract.code_execution.model_dump(mode='json'),
        }
        if adaptive:
            # One trajectory, one schedule horizon: 5k/20k/etc are real prefixes.
            train_request["schedule_steps"] = contract.rungs[-1]
        request_hash, series = training_binding(train_request), series_binding(train_request)
        identity = {
            'train': train_request,
            'env': case.environment,
            'eval_seeds': eval_seeds,
            'case': case.name,
            'candidate': action.candidate,
            'task': self.task.name,
            'purpose': purpose,
        }
        tid = ("check-" if check else "t-") + digest(identity)[:24]
        # 实验 3｜完全相同的已完成请求只在工件校验通过后复用。
        existing = self.store.trial(tid)
        if existing and existing["status"] == "complete":
            # Cache reuse is also checked, not blindly trusted because a row exists.
            current = self.root / "jobs" / tid
            trained = read_json(current / "train/training.json")
            if (trained["request_sha256"] != request_hash or
                    directory_digest(current / "train/artifact") != trained["artifact_sha256"]):
                raise ValueError("Completed trial training artifact changed")
            evaluated = read_json(current / "evaluation/evaluation.json")
            if evaluated["artifact_sha256"] != trained["artifact_sha256"]:
                raise ValueError("Completed trial evaluation binding changed")
            return existing
        job = self.root / "jobs" / tid
        job.mkdir(parents=True, exist_ok=True)
        training_done = (job / "train/training.json").exists()
        evaluation_done = (job / "evaluation/evaluation.json").exists()
        # 实验 4｜区分同一请求恢复、较早端点续训、同端点仅增加评价。
        training_reuse = None
        resumed_from = None
        start_step = 0
        read_only = [Path(case.data)]
        request_path = job / "train-request.json"
        if request_path.exists():
            prior = read_json(request_path)
            if training_binding(prior) != request_hash:
                raise ValueError("Existing experiment request changed")
            train_request = prior  # Keep the same continuation receipt on retries.
            resumed_from = prior.get("resume_from", {}).get("trial")
            start_step = int(prior.get("resume_from", {}).get("steps", 0))
            if prior.get("resume_from"):
                read_only.append(Path(prior["resume_from"]["path"]))
        elif adaptive and not training_done:
            # Same candidate/data/seed: prefer the latest compatible checkpoint.
            # Different environments/seeds/algorithms always start new training.
            rows = [r for r in self.store.trials() if r["candidate"] == action.candidate
                    and r["case"] == case.name and r["train_seed"] == action.train_seed
                    and r["steps"] <= action.steps and r["trial_id"] != tid]
            if continuation is not None:
                rows = [r for r in rows if r["trial_id"] == continuation["trial_id"]]
            for row in sorted(rows, key=lambda r: r["steps"], reverse=True):
                receipt = self._training_receipt(row, series)
                if receipt is None:
                    continue
                path, trained = receipt
                if row["steps"] == action.steps:
                    training_reuse = (row, path, trained)
                else:
                    resumed_from, start_step = row["trial_id"], row["steps"]
                    train_request["resume_from"] = {
                        'trial': row['trial_id'],
                        'path': str(path / 'checkpoint.pt'),
                        'sha256': trained['checkpoint_sha256'],
                        'steps': start_step,
                        'series_sha256': series,
                        'request_sha256': trained['request_sha256'],
                    }
                    read_only.append(path / "checkpoint.pt")
                break
            if continuation is not None and training_reuse is None and not resumed_from:
                raise ValueError("Source checkpoint is not compatible with this trajectory")
        # 实验 5｜保存请求并预留本次尝试资源。
        # 新增训练按累计端点差计算；未知失败消耗不会在恢复时自动清零。
        write_json(request_path, train_request)
        count_key = "attempt_count:" + tid
        attempts = int(self.store.get(count_key, 0))
        if attempts >= contract.budget.max_attempts_per_trial:
            raise BudgetExceeded("Trial retry limit reached; no automatic budget reset")
        incremental = 0 if training_done or training_reuse else action.steps - start_step
        attempt = tid + "-" + uuid.uuid4().hex[:8]
        self.store.reserve(
            attempt,
            {
                'attempts': 1,
                'checks' if check else 'trials': 0 if existing else 1,
                'train_steps': incremental,
                'optimizer_steps': incremental * optimizer_bound,
                'eval_episodes': 0 if evaluation_done else len(eval_seeds),
            },
            limits(contract),
        )
        self.store.put(count_key, attempts + 1)
        pending = {
            'purpose': purpose,
            'trial_id': tid,
            'candidate': action.candidate,
            'case': case.name,
            'train_seed': action.train_seed,
            'steps': action.steps,
            'attempt': attempt,
            'eval_seeds': eval_seeds,
            'continued_from': resumed_from,
            'start_step': start_step,
            'schedule_steps': train_request.get('schedule_steps', action.steps),
        }
        self.store.save_trial(tid, action.candidate, "running", pending)
        # 实验 6｜准备或恢复训练，再由独立评价器产生实际回报。
        try:
            if training_reuse:
                row, source, _ = training_reuse
                destination = job / "train"
                destination.mkdir(exist_ok=True)
                for relative in ("artifact", "checkpoint.pt", "training.json", "metrics.jsonl"):
                    path = source / relative
                    if not path.exists():
                        continue
                    if path.is_symlink() or (path.is_dir() and any(x.is_symlink() for x in path.rglob("*"))):
                        raise ValueError("Cannot reuse a symlinked candidate artifact")
                    if path.is_dir():
                        shutil.copytree(path, destination / relative, dirs_exist_ok=True)
                    else:
                        shutil.copy2(path, destination / relative)
                training_done = True
            if not training_done:
                self._worker("train", request_path, job / "train", read_only)
            trained = read_json(job / "train/training.json")
            if trained["request_sha256"] != request_hash or trained["steps"] != action.steps:
                raise ValueError("Worker training result violates request binding")
            policy = job / "train/artifact"
            if code_candidate and digest(read_json(policy / "algorithm.json")) != digest(recipe):
                raise ValueError("Exported candidate source/config differs from requested algorithm")
            if trained.get("optimizer_steps", 0) > action.steps * optimizer_bound:
                raise ValueError("Candidate exceeded its optimizer-step reservation")
            if directory_digest(policy) != trained["artifact_sha256"]:
                raise ValueError("Policy content changed after training")
            # 评分请求绑定刚验证的策略工件；不接受候选打印的自报分数。
            eval_request = {
                'artifact': str(policy),
                'artifact_sha256': trained['artifact_sha256'],
                'training_binding': request_hash,
                'environment': case.environment,
                'eval_seeds': eval_seeds,
                'steps': action.steps,
                'source_sha256': self.store.get('source_sha256'),
                'code_execution': contract.code_execution.model_dump(mode='json'),
            }
            write_json(job / "eval-request.json", eval_request)
            if not evaluation_done:
                self._worker("evaluate", job / "eval-request.json", job / "evaluation", [policy])
            evaluated = read_json(job / "evaluation/evaluation.json")
            if evaluated["evaluation_request_sha256"] != digest(eval_request):
                raise ValueError("Independent evaluation binding mismatch")
            if evaluated["artifact_sha256"] != trained["artifact_sha256"]:
                raise ValueError("Independent evaluator scored a different policy")
            if len(evaluated["returns"]) != len(eval_seeds):
                raise ValueError("Evaluator returned the wrong number of episodes")
            result = TrialResult(purpose=purpose, trial_id=tid, candidate=action.candidate, case=case.name, train_seed=action.train_seed, steps=action.steps, returns=evaluated['returns'], score=evaluated['score'], metric=evaluated['metric'], artifact_sha256=trained['artifact_sha256'], request_sha256=request_hash, train_seconds=trained['train_seconds'], evaluate_seconds=evaluated['evaluate_seconds'], diagnostics=trained['diagnostics'], evaluation_kind=evaluated['evaluation_kind'], optimizer_steps=trained.get('optimizer_steps', action.steps * optimizer_bound), value_learning_audit=trained.get('value_learning_audit', {}), inference=evaluated.get('inference', {}), schedule_steps=train_request.get('schedule_steps', action.steps), series_sha256=trained.get('series_sha256', series), source_sha256=train_request['source_sha256'], checkpoint_sha256=trained.get('checkpoint_sha256', ''), continued_from=resumed_from, training_reused_from=training_reuse[0]['trial_id'] if training_reuse else None, resumed_at=trained['resumed_at'], incremental_train_steps=incremental, eval_seeds=eval_seeds).model_dump(
                mode='json',
            )
            # 实验 7｜事务提交结果、可见范围和已知实际用量。
            # 失败分支保存诊断后重新抛出，交给 Harness 决定后续行动。
            with self.store.transaction():
                self.store.save_trial(tid, action.candidate, "complete", result)
                self.store.event(
                    'trial_complete',
                    {
                        'trial_id': tid,
                        'score': result['score'],
                        'split': authorized_split,
                    },
                )
                self.store.expose(
                    tid,
                    observer or ('researcher' if authorized_split == 'dev' else 'independent_auditor'),
                    authorized_split,
                )
                self.store.record_usage(
                    attempt,
                    {
                        'train_steps': 0 if training_done else action.steps - trained['resumed_at'],
                        'eval_episodes': 0 if evaluation_done else len(evaluated['returns']),
                        'optimizer_steps': (
                            0
                            if training_done
                            else trained.get('optimizer_steps', action.steps * optimizer_bound) - trained.get('resumed_optimizer_steps', trained['resumed_at'] * optimizer_bound)
                        ),
                        'train_seconds': 0 if training_done else trained['train_seconds'],
                        'evaluation_seconds': 0 if evaluation_done else evaluated['evaluate_seconds'],
                    },
                )
            return {"status": "complete", **result}
        except Exception as exc:
            self.store.save_trial(
                tid,
                action.candidate,
                'failed',
                {
                    **pending,
                    'error': str(exc)[-4500:],
                    'error_type': type(exc).__name__,
                    'diagnostics_tool': {
                        'kind': 'read_trial_log',
                        'trial': tid,
                        'stream': 'train',
                        'tail': True,
                    },
                },
            )
            self.store.put("last_failed_trial", tid)
            self.store.event("trial_failed", {**pending, "error": str(exc)[-4500:]})
            raise
