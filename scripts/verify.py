from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class VerificationCheck:
    name: str
    argv: tuple[str, ...]


CommandRunner = Callable[[tuple[str, ...], Path], int]


def verification_checks(*, include_install: bool = True) -> tuple[VerificationCheck, ...]:
    checks = (
        VerificationCheck(
            "install",
            (sys.executable, "-m", "pip", "install", "-e", ".[dev]"),
        ),
        VerificationCheck("lint", (sys.executable, "-m", "ruff", "check", ".")),
        VerificationCheck("typecheck", (sys.executable, "-m", "mypy", "src", "tests", "scripts")),
        VerificationCheck("determinism", (sys.executable, "scripts/check_determinism.py")),
        VerificationCheck("test", (sys.executable, "-m", "pytest", "-q")),
        VerificationCheck("build", (sys.executable, "-m", "build")),
        VerificationCheck(
            "evaluation_suite",
            (
                sys.executable,
                "-c",
                (
                    "from dsg_spatialqa_lab import run_evaluation_suite; "
                    "suite = run_evaluation_suite(); "
                    "summary = suite['summary']; "
                    "assert summary['failed'] == 0, summary; "
                    "print(summary); "
                    "print(suite['digest'])"
                ),
            ),
        ),
    )
    if include_install:
        return checks
    return checks[1:]


def run_verification(
    checks: Sequence[VerificationCheck],
    *,
    cwd: Path,
    runner: CommandRunner | None = None,
) -> int:
    command_runner = _run_command if runner is None else runner
    for check in checks:
        print(f"[verify] {check.name}: {' '.join(check.argv)}", flush=True)
        result = command_runner(check.argv, cwd)
        if not isinstance(result, int):
            raise TypeError("runner must return an integer exit code")
        if result != 0:
            print(f"[verify] failed: {check.name} exited with {result}", file=sys.stderr)
            return result
    print("[verify] all checks passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run DSG-SpatialQA Lab verification gates.")
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip the editable dev install when dependencies are already present.",
    )
    args = parser.parse_args(argv)
    return run_verification(
        verification_checks(include_install=not args.skip_install),
        cwd=ROOT,
    )


def _run_command(argv: tuple[str, ...], cwd: Path) -> int:
    return subprocess.run(argv, cwd=cwd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
