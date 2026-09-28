"""Real training + independent policy inference, with reviewed teaching decisions."""
from pathlib import Path

from arlab.cli import demo_contract
from arlab.core.harness import Harness


def main() -> None:
    root = Path("runs/open-lesson")
    contract = demo_contract(root, steps=40, candidates=3, open_code=True)
    agent = Harness(contract, root / "campaign")
    try:
        report = agent.run()
        print(report["status"], report["trial_counts"])
        print("Synthetic smoke results only, not D4RL performance evidence.")
    finally:
        agent.close()


if __name__ == "__main__":
    main()
