"""Small terminal trace of actual decisions, not private chain-of-thought."""
from __future__ import annotations


def _brief(value, limit: int) -> str:
    """Render model-authored plan fields safely as compact terminal text."""
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


class ProgressModel:
    def __init__(self, model):
        self.model = model
        self.count = 0

    def complete(self, instruction, context, schema):
        reply = self.model.complete(instruction, context, schema)
        self.count += 1
        action = reply.payload.get("action", {})
        details = ", ".join(f"{k}={action[k]}" for k in ("case", "train_seed", "steps") if k in action)
        print(f"[行动 {self.count}] {action.get('kind', 'invalid')} {details}", flush=True)
        if action.get("kind") == "plan":
            print(f"  核心思路：{_brief(action.get('reason'), 300)}", flush=True)
            items = action.get("items", [])
            for index, item in enumerate(items[:5], 1):
                print(f"  {index}. {_brief(item.get('objective'), 200)}", flush=True)
            if len(items) > 5:
                print(f"  ……另有 {len(items) - 5} 项", flush=True)
        elif action.get("kind") == "review_proposal":
            print(f"  [提案] {_brief(action.get('hypothesis', {}).get('question'), 300)}", flush=True)
            print(f"  方法变化：{_brief(action.get('method_change'), 400)}", flush=True)
            print(f"  非调参理由：{_brief(action.get('why_not_tuning'), 300)}", flush=True)
            print(f"  可行性：{_brief(action.get('feasibility'), 300)}", flush=True)
            print(f"  失败风险：{_brief(action.get('failure_mode'), 300)}", flush=True)
            print(f"  自评结论：{_brief(action.get('verdict'), 30)}", flush=True)
        elif action.get("kind") in ("propose", "edit_algorithm"):
            print(f"  [执行提案] {_brief(action.get('hypothesis', {}).get('question'), 300)}", flush=True)
            print(f"  预期：{_brief(action.get('hypothesis', {}).get('prediction'), 300)}", flush=True)
            print(f"  证伪条件：{_brief(action.get('hypothesis', {}).get('falsification'), 300)}", flush=True)
        return reply

    def feedback(self, action, feedback):
        """Print bounded public tool outcomes, never model hidden reasoning."""
        action = action.model_dump(mode="json") if hasattr(action, "model_dump") else action
        feedback = feedback.model_dump(mode="json") if hasattr(feedback, "model_dump") else feedback
        details = feedback.get("details", {})
        if details.get("loop_guard"):
            print(f"  [循环保护] {_brief(feedback.get('summary'), 360)}", flush=True)
            return
        if action.get("kind") in ("review_proposal", "propose", "edit_algorithm"):
            print(f"  [提案反馈] {_brief(feedback.get('summary'), 360)}", flush=True)
        kind = action.get("kind")
        if kind == "inspect":
            location = f"{action.get('target')}:{action.get('id') or '-'}"
            lines = f"{details.get('start_line', '?')}-{details.get('end_line', '?')}"
            print(f"  [读取] {location}，行 {lines}/{details.get('total_lines', '?')}，"
                  f"后续页={'有' if details.get('next') else '无'}", flush=True)
        elif kind == "read_trial_log":
            metrics = ", ".join(list(details.get("metric_summary", {}))[:8]) or "无数值摘要"
            print(f"  [日志] {action.get('trial')} {action.get('stream')}，"
                  f"{details.get('filtered_lines', 0)} 行；指标：{metrics}", flush=True)
        elif kind.startswith("search_"):
            print(f"  [检索] 返回 {len(details.get('items', []))} 项；"
                  f"后续偏移={details.get('next_offset', '无')}", flush=True)
        elif kind in ("run", "continue_trial", "check_candidate"):
            score = details.get("score")
            print(f"  [实验] trial={details.get('trial_id', '-')}，"
                  f"score={score if score is not None else '未提供'}", flush=True)
