"""Run the same real IQL classroom example using the public Python API."""
from pathlib import Path

from arlab.cli import demo_contract
from arlab.core.harness import Harness


def main() -> None:
    root = Path("runs/python-lesson")
    contract = demo_contract(root, steps=80, candidates=3)
    harness = Harness(contract, root / "campaign")
    try:
        report = harness.run()
        print(report["status"])
        print(root / "campaign" / "report.md")
    finally:
        harness.close()


if __name__ == "__main__":
    main()
