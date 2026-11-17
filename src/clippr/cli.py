"""Command line: ``clippr run``, ``clippr worker``, ``clippr serve``."""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from pathlib import Path

from clippr.config import get_settings


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "faster_whisper", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def cmd_run(args: argparse.Namespace) -> int:
    from clippr.pipeline.moments import PICKER_LABELS
    from clippr.pipeline.runner import overall_progress, run_pipeline

    settings = get_settings()
    if args.selector:
        settings = settings.model_copy(update={"selector": args.selector})
    out = Path(args.out) if args.out else settings.jobs_dir / f"cli-{uuid.uuid4().hex[:8]}"
    tty = sys.stdout.isatty()
    last = {"line": "", "msg": ""}

    def report(stage: str, frac: float, msg: str) -> None:
        pct = int(overall_progress(stage, frac) * 100)
        line = f"[{pct:3d}%] {msg}"
        if tty and line != last["line"]:
            print("\r" + line.ljust(70), end="", flush=True)
        elif not tty and msg != last["msg"]:  # in logs: one line per new message
            print(line, flush=True)
        last["line"], last["msg"] = line, msg

    result = run_pipeline(args.source, out, settings, max_clips=args.clips or settings.max_clips,
                          focus=args.focus, report=report)
    print()
    print(f"\nPicked by: {PICKER_LABELS.get(result.selected.picker, result.selected.picker)}")
    for note in result.selected.notes:
        print(f"Note: {note}")
    for c in result.clips:
        m = c.moment
        print(f"\n#{c.index}  {m.title}  [{m.start_s:.1f}s – {m.end_s:.1f}s, score {m.score}]")
        print(f"    hook: {m.hook}")
        print(f"    {c.video}")
    return 0


def cmd_worker(args: argparse.Namespace) -> int:
    from clippr.worker import worker_loop

    worker_loop(get_settings(), once=args.once)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from clippr.web.app import create_app

    uvicorn.run(create_app(get_settings()), host=args.host, port=args.port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clippr", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Make clips from a YouTube link or a local video file")
    run.add_argument("source", help="YouTube link or path to a video file")
    run.add_argument("-n", "--clips", type=int, help="number of clips (default from settings)")
    run.add_argument("--focus", help="what the clips should be about, e.g. 'the funniest moments'")
    run.add_argument("--selector", choices=["auto", "claude", "local", "rules"])
    run.add_argument("--out", help="output directory (default: data/jobs/cli-<id>)")
    run.set_defaults(func=cmd_run)

    worker = sub.add_parser("worker", help="Run the background worker that processes web jobs")
    worker.add_argument("--once", action="store_true", help="exit when the queue is empty")
    worker.set_defaults(func=cmd_worker)

    serve = sub.add_parser("serve", help="Run the web app")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except Exception as e:  # noqa: BLE001 - show a clean message, details with -v
        if args.verbose:
            raise
        print(f"\nError: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
