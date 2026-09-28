#!/usr/bin/env python3
"""Compare this reading edition with its unmodified source ZIP (standard library only).

Usage:
    python scripts/verify_readability.py --baseline /path/autoresearch-agent-v0.6.0.zip

This is an owner-invoked verification utility, not an Agent tool. It parses source
without importing or executing baseline code. Matching ASTs do not imply equal
source hashes, traceback line numbers, timing, or arbitrary reflective behavior.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_baseline(path: Path) -> dict[str, bytes]:
    """Read an archive in memory; never extract user-supplied paths."""
    with zipfile.ZipFile(path) as archive:
        entries = [entry for entry in archive.infolist() if not entry.is_dir()]
        roots = {PurePosixPath(entry.filename).parts[0] for entry in entries}
        if len(roots) != 1:
            raise ValueError("Expected one top-level source directory in the baseline ZIP")
        result: dict[str, bytes] = {}
        total = 0
        for entry in entries:
            name = PurePosixPath(entry.filename)
            if name.is_absolute() or ".." in name.parts or len(name.parts) < 2:
                raise ValueError(f"Unsafe archive member: {entry.filename}")
            total += entry.file_size
            if entry.file_size > 20_000_000 or total > 100_000_000:
                raise ValueError("Baseline archive exceeds verification size limit")
            relative = str(PurePosixPath(*name.parts[1:]))
            if relative in result:
                raise ValueError(f"Duplicate archive member: {relative}")
            result[relative] = archive.read(entry)
        return result


def syntax(data: bytes) -> str:
    """Include all expressions, docstrings, annotations and type comments."""
    return ast.dump(ast.parse(data, type_comments=True), include_attributes=False)


def protected(relative: str) -> bool:
    """Assets affecting research behavior that must be byte-for-byte unchanged."""
    if relative == "configs/README.md":
        return False  # Human-facing documentation, not configuration data.
    roots = (
        "tests/", "examples/", "configs/", "requirements/", "docker/", ".github/",
        "src/arlab/algorithms/", "src/arlab/tasks/", "src/arlab/templates/",
        "src/arlab/skills/", "src/arlab/research/", "src/arlab/deployment/",
    )
    files = {
        "start.sh", "deploy.sh", "pyproject.toml", "MANIFEST.in",
        "src/arlab/core/config.py", "src/arlab/core/schema.py",
        "src/arlab/core/tool_catalog.py", "src/arlab/core/presets.py",
        "src/arlab/core/models.py", "src/arlab/core/io.py",
    }
    return relative.startswith(roots) or relative in files


def verify(baseline: Path, target: Path) -> dict:
    original = load_baseline(baseline)
    ast_failures, missing, protected_failures, changed_python = [], [], [], []
    python_count = package_count = protected_count = 0
    baseline_package, target_package = {}, {}
    for relative, before in sorted(original.items()):
        path = target / relative
        if relative.endswith(".py"):
            python_count += 1
            if not path.is_file() or path.is_symlink():
                missing.append(relative)
                continue
            after = path.read_bytes()
            if before != after:
                changed_python.append(relative)
            if syntax(before) != syntax(after):
                ast_failures.append(relative)
            if relative.startswith("src/arlab/"):
                package_count += 1
                package_key = relative.removeprefix("src/arlab/")
                baseline_package[package_key] = sha256(before)
                target_package[package_key] = sha256(after)
        if protected(relative):
            protected_count += 1
            if not path.is_file() or path.is_symlink() or path.read_bytes() != before:
                protected_failures.append(relative)
    def source_identity(mapping: dict) -> str:
        value = json.dumps(mapping, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return sha256(value.encode())
    baseline_runtime = {p for p in original if p.startswith("src/arlab/")}
    target_runtime = {
        str(p.relative_to(target)) for p in (target / "src/arlab").rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }
    new_runtime = sorted(target_runtime - baseline_runtime)
    missing_runtime = sorted(baseline_runtime - target_runtime)
    passed = not any((ast_failures, missing, protected_failures, new_runtime, missing_runtime))
    return {
        "passed": passed,
        "baseline_zip_sha256": sha256(baseline.read_bytes()),
        "baseline_python_files": python_count,
        "package_python_files": package_count,
        "changed_python_files": changed_python,
        "ast_equal_including_docstrings_and_types": not ast_failures and not missing,
        "ast_failures": ast_failures,
        "missing_python_files": missing,
        "protected_files_checked": protected_count,
        "protected_byte_mismatches": protected_failures,
        "new_runtime_files": new_runtime,
        "missing_runtime_files": missing_runtime,
        "baseline_source_sha256": source_identity(baseline_package),
        "reading_edition_source_sha256": source_identity(target_package),
        "scope": "Existing program syntax/constant preservation, not bytecode or timing identity",
        "caveat": "Source hashes and traceback locations change; do not resume old campaigns under this edition",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--target", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.baseline, args.target.resolve())
    except (OSError, ValueError, SyntaxError, zipfile.BadZipFile) as exc:
        parser.exit(2, f"Verification failed: {exc}\n")
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
