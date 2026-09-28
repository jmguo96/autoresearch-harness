"""Evidence-first classroom report; never substitute a development score for a final one."""
from __future__ import annotations

import csv
import io
import statistics
from pathlib import Path

from arlab.core.io import atomic_write, read_json, write_json
from arlab.core.schema import ResearchContract
from arlab.core.stats import paired_comparison


def study_report(root: Path, contract: ResearchContract) -> dict:
    final_spec = contract.final_evaluation
    if final_spec is None:
        raise ValueError("The full study requires a predeclared final panel")
    development = read_json(root / "development/report.json") if (root / "development/report.json").exists() else {}
    audit_path = root / "final/audit.json"
    final = read_json(audit_path) if audit_path.exists() else {}
    progress_path = root / "final/progress.json"
    progress = read_json(progress_path) if progress_path.exists() else {}
    trials = final.get("trials", [])
    baseline = [t for t in trials if t["candidate"] == final.get("baseline_candidate")]
    selected = [t for t in trials if t["candidate"] == final.get("selected_candidate")]
    expected = {(name, seed, final_spec.steps) for name in final_spec.cases for seed in final_spec.train_seeds}
    comparison = paired_comparison(baseline, selected, expected_cells=expected)
    complete = bool(final.get("status") == "completed" and comparison["contract_panel_complete"])
    state = "final_evaluation_complete" if complete else (
        "final_evaluation_pending" if development.get("status") == "completed" else "exploration_incomplete")
    rows = []
    for name in final_spec.cases:
        b = [t for t in baseline if t["case"] == name]
        c = [t for t in selected if t["case"] == name]
        rows.append({"task": name, "used_for_development": any(x.name == name and x.split == "dev" for x in contract.cases),
                     "baseline_seeds": len(b), "candidate_seeds": len(c),
                     "expected_seeds": len(final_spec.train_seeds),
                     "baseline_mean": statistics.mean(t["score"] for t in b) if complete else None,
                     "baseline_std": statistics.stdev(t["score"] for t in b) if complete and len(b) > 1 else None,
                     "candidate_mean": statistics.mean(t["score"] for t in c) if complete else None,
                     "candidate_std": statistics.stdev(t["score"] for t in c) if complete and len(c) > 1 else None,
                     "delta": statistics.mean(t["score"] for t in c) - statistics.mean(t["score"] for t in b) if complete else None})
    development_tasks = [row["task"] for row in rows if row["used_for_development"]]
    not_used_for_search = [row["task"] for row in rows if not row["used_for_development"]]
    heldout_expected = {x for x in expected if x[0] in not_used_for_search}
    heldout = paired_comparison([r for r in baseline if r["case"] in not_used_for_search],
                                [r for r in selected if r["case"] in not_used_for_search],
                                expected_cells=heldout_expected)
    independent_positive = bool(complete and heldout_expected and
                                heldout["contract_panel_complete"] and heldout["status"] == "positive_evidence")
    all_d4rl = all(c.environment.get("kind") == "d4rl" for c in contract.cases)
    report = {"status": state, "research_complete": complete, "final_conclusion_available": complete,
              "scope": "D4RL-nine-task" if all_d4rl and len(final_spec.cases) == 9 else "protocol_test_NOT_D4RL",
              "development_status": development.get("status", "not_started"),
              "final_progress": progress, "required_tasks": len(final_spec.cases),
              "final_steps": final_spec.steps, "training_seeds": final_spec.train_seeds,
              "rows": rows, "comparison": comparison if complete else None,
              "development_tasks_in_final": development_tasks,
              "not_used_for_search_tasks": not_used_for_search,
              "development_task_comparison": final.get("development_task_comparison") if complete else None,
              "unseen_task_comparison": final.get("unseen_task_comparison") if complete else None,
              "all_tasks_descriptive_gain_positive": bool(complete and comparison.get("macro_delta", 0) > 0),
              "all_tasks_positive_paired_evidence": bool(complete and comparison["status"] == "positive_evidence"),
              "not_used_for_search_evidence": heldout if complete else None,
              "not_used_for_search_improvement_supported": independent_positive,
              "method_improved": independent_positive,
              "method_improved_scope": "Deprecated alias of not_used_for_search_improvement_supported; not a universal claim",
              "note": "Complete evidence is not guaranteed improvement. Developed tasks are not held-out; "
                      "the all-task table is descriptive and the unseen-task panel is reported separately. "
                      "Final outcomes never feed back to the already frozen researcher.",
              "inference": "No LLM was called to fill or grade this results table."}
    write_json(root / "report.json", report)
    metric = "D4RL 归一化分数" if all_d4rl else "合成协议测试回报（不是 D4RL）"
    lines = ["# Autoresearch 研究报告", "", f"状态：**{state}**", "",
             f"最终要求：{len(rows)} 个任务 × {len(final_spec.train_seeds)} 个训练种子，"
             f"每次 {final_spec.steps:,} 更新；固定 IQL 与冻结候选使用匹配条件。", "",
             "**证据不完整；开发任务探索不能替代完整九任务最终结论。**" if not complete else
             "**最终实验面板完整。是否提高、在哪些任务提高，需要分别解释，不能只看 completed。**", "",
             f"证据范围：{len(development_tasks)} 个曾用于开发的任务，"
             f"{len(not_used_for_search)} 个未用于本次搜索的任务；换新种子不会重置开发历史。", "",
             f"指标：{metric}；± 表示不同训练种子间的标准差，不是标准误。", "",
             "| 任务 | 搜索时使用 | IQL mean ± std | 冻结候选 mean ± std | 差值 |",
             "|---|---|---:|---:|---:|"]
    for row in rows:
        def value(prefix):
            mean, sd = row[prefix + "_mean"], row[prefix + "_std"]
            if mean is None:
                return "未形成完整最终结果"
            return f"{mean:.3f} ± {sd:.3f}" if sd is not None else f"{mean:.3f}"
        delta = "—" if row["delta"] is None else f"{row['delta']:+.3f}"
        lines.append(f"| {row['task']} | {'是' if row['used_for_development'] else '否'} | "
                     f"{value('baseline')} | {value('candidate')} | {delta} |")
    if complete:
        for label, key in (("开发任务复验", "development_task_comparison"),
                           ("未用于本次搜索任务", "unseen_task_comparison")):
            group = final.get(key) or {}
            if group.get("contract_panel_complete") and group.get("macro_delta") is not None:
                lines += ["", f"{label}平均差值：{group['macro_delta']:+.3f}。"]
        lines += ["", f"未用于搜索任务的改善证据：{'获得当前配对证据支持' if independent_positive else '尚未获得充分支持'}。",
                  f"全任务描述性平均差值：{comparison['macro_delta']:+.3f}。",
                  "逐任务结果、区间与其余未用于搜索任务的单独比较见 report.json。",
                  "最终选择为 IQL 本身时只执行一次基线面板，不制造重复实验或改善声明。"]
    lines += ["", "## 解释边界", "",
              "同一个 D4RL 开发任务换新种子并不成为未见任务。九项中开发任务单独标明。",
              "固定的是方法源码与配置；各任务必须用自己的离线数据重新训练，不是一份开发任务权重跨任务评价。",
              "等更新次数不等于等 FLOPs 或推理开销；原始审计记录保留实际训练和推理成本。",
              "看完最终结果再修改方法，会消耗这组结果的独立性；改名或换种子不能恢复它。",
              "合成环境测试、少量更新与测试替身不能写成 D4RL 研究结果。"]
    atomic_write(root / "report.md", "\n".join(lines) + "\n")
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(root / "results.csv", out.getvalue())
    return report
