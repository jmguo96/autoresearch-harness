from types import SimpleNamespace

from arlab.classroom.output import ProgressModel


class StaticModel:
    def __init__(self, action):
        self.action = action

    def complete(self, instruction, context, schema):
        return SimpleNamespace(payload={"summary": "fixture", "action": self.action})


def test_progress_prints_plan_core_idea_and_objectives(capsys):
    model = ProgressModel(StaticModel({
        "kind": "plan",
        "reason": "Compare the baseline\nagainst one controlled change.",
        "items": [
            {"id": "baseline", "objective": "Measure the baseline."},
            {"id": "candidate", "objective": "Test the candidate."},
        ],
    }))

    model.complete("instruction", {}, object)

    output = capsys.readouterr().out
    assert "[行动 1] plan" in output
    assert "核心思路：Compare the baseline against one controlled change." in output
    assert "1. Measure the baseline." in output
    assert "2. Test the candidate." in output


def test_progress_keeps_non_plan_output_compact(capsys):
    action = {"kind": "run", "case": "task", "train_seed": 0, "steps": 5000}
    model = ProgressModel(StaticModel(action))

    model.complete("instruction", {}, object)

    assert capsys.readouterr().out == "[行动 1] run case=task, train_seed=0, steps=5000\n"


def test_progress_prints_proposal_and_self_review(capsys):
    action = {"kind": "review_proposal", "parent": "c-base",
              "hypothesis": {"question": "Does a new actor objective help?"},
              "method_change": "Replace weighted cloning with a conservative objective.",
              "why_not_tuning": "The optimization objective changes.",
              "feasibility": "Implement in candidate source.",
              "failure_mode": "May underfit.", "verdict": "proceed"}
    model = ProgressModel(StaticModel(action))
    model.complete("instruction", {}, object)
    model.feedback(action, {"status": "ok", "summary": "Proposal self-review: proceed.",
                            "details": {"review_id": "event-1"}})
    output = capsys.readouterr().out
    assert "[提案] Does a new actor objective help?" in output
    assert "方法变化：Replace weighted cloning" in output
    assert "非调参理由：The optimization objective changes." in output
    assert "自评结论：proceed" in output
    assert "[提案反馈] Proposal self-review: proceed." in output


def test_progress_prints_read_result_and_loop_guard(capsys):
    model = ProgressModel(StaticModel({"kind": "inspect", "target": "candidate"}))
    model.feedback(
        {"kind": "inspect", "target": "candidate", "id": "c-1"},
        {"status": "ok", "summary": "read", "details": {
            "start_line": 1, "end_line": 20, "total_lines": 20, "next": None,
        }},
    )
    model.feedback(
        {"kind": "inspect", "target": "candidate", "id": "c-1"},
        {"status": "invalid_action", "summary": "Use retained evidence.",
         "details": {"loop_guard": True}},
    )
    output = capsys.readouterr().out
    assert "[读取] candidate:c-1，行 1-20/20，后续页=无" in output
    assert "[循环保护] Use retained evidence." in output
