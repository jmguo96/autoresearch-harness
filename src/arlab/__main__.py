"""CLI process entry: let cancellation unwind worker cleanup handlers."""
import signal

from arlab.cli import main


def interrupt(_signum, _frame):
    raise KeyboardInterrupt


signal.signal(signal.SIGTERM, interrupt)
raise SystemExit(main())
