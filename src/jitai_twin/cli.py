"""Command-line interface for the synthetic demo and release audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import DemoConfig, run_synthetic_demo
from .release_audit import audit_release


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jitai-twin")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run the synthetic-only pipeline")
    demo.add_argument("--output", type=Path, default=Path("outputs/demo"))
    demo.add_argument("--quick", action="store_true", help="use the CI-sized configuration")
    demo.add_argument("--seed", type=int, default=19)
    demo.add_argument("--device", default="cpu")

    audit = subparsers.add_parser("audit", help="check a tree for private release material")
    audit.add_argument("--root", type=Path, default=Path("."))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "demo":
        config = DemoConfig.quick(args.seed) if args.quick else DemoConfig(seed=args.seed)
        summary = run_synthetic_demo(args.output, config, device=args.device)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    issues = audit_release(args.root)
    if issues:
        for issue in issues:
            print(f"FAIL {issue.path}: {issue.reason}")
        return 1
    print(f"PASS: no release-policy violations under {args.root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
