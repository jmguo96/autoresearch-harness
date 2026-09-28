#!/usr/bin/env python3
"""Small maintenance checks, not a substitute for tests or a security proof."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    errors: list[str] = []
    files = sorted((ROOT / "src").rglob("*.py"))
    statements = 0
    for path in files:
        relative = path.relative_to(ROOT)
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text, filename=str(relative))
            compile(tree, str(relative), "exec")
        except SyntaxError as exc:
            errors.append(f"{relative}: {exc}")
            continue
        statements += sum(isinstance(node, ast.stmt) for node in ast.walk(tree))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {"eval", "exec"}:
                    errors.append(f"{relative}:{node.lineno}: dynamic {node.func.id} is forbidden")
            if "core" in relative.parts:
                imports = []
                if isinstance(node, ast.ImportFrom):
                    imports = [node.module or ""]
                elif isinstance(node, ast.Import):
                    imports = [alias.name for alias in node.names]
                if any(name in {"gym", "d4rl", "gymnasium"} or
                       name.startswith("arlab.tasks.iql") for name in imports):
                    errors.append(f"{relative}:{node.lineno}: domain dependency leaked into core")
    print(json.dumps({"python_files": len(files), "statements": statements,
                      "errors": errors, "scope": "syntax and explicit architecture checks only"}, indent=2))
    return int(bool(errors))


if __name__ == "__main__":
    sys.exit(main())
