"""A real subprocess entry point for training and independently scoring artifacts."""
from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from arlab.core.io import read_json, source_digest, write_json
from arlab.tasks.protocol import get_task


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "evaluate"])
    parser.add_argument("--task", required=True)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        task = get_task(args.task)
        request = read_json(args.request)
        if request.get("source_sha256") and request["source_sha256"] != source_digest():
            raise ValueError("Worker implementation differs from the frozen request")
        result = task.train(request, args.output) if args.mode == "train" else task.evaluate(request, args.output)
        print(f"{args.mode} completed: {result.get('steps', result.get('score'))}", flush=True)
        return 0
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / "error.json", {"type": type(exc).__name__, "message": str(exc)[:4000],
            "traceback": traceback.format_exc()[-12000:]})
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import signal

    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    raise SystemExit(main())
