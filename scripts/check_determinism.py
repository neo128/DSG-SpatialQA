from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = (
    ROOT / ".github",
    ROOT / "scripts",
    ROOT / "src",
    ROOT / "tests",
)
CHECKED_SUFFIXES = frozenset((".py", ".toml", ".yaml", ".yml"))
SKIPPED_PARTS = frozenset(
    (
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "build",
        "dist",
    )
)


def _join(*parts: str) -> str:
    return "".join(parts)


BLOCKED_TERMS = (
    (_join("date", "time"), "clock_date_time"),
    (_join("time", "."), "clock_time_module"),
    (_join("time", "("), "clock_time_call"),
    (_join("rand", "om"), "rand_source"),
    (_join("req", "uests"), "net_req_pkg"),
    (_join("url", "lib"), "net_url_pkg"),
    (_join("http", "x"), "net_http_client"),
    (_join("open", "ai"), "model_api_pkg"),
    (_join("sock", "et"), "net_sock_pkg"),
)


def scan_paths(paths: Sequence[Path]) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for file_path in _iter_checked_files(paths):
        for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), 1):
            for term, reason in BLOCKED_TERMS:
                if term in line:
                    matches.append(
                        {
                            "path": _display_path(file_path),
                            "line": line_number,
                            "pattern": term,
                            "reason": reason,
                        }
                    )
    return {"valid": not matches, "matches": matches}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scan project source files for nondeterministic runtime boundaries."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Explicit files or directories to scan. Defaults to project source paths.",
    )
    args = parser.parse_args(argv)
    paths = tuple(args.paths) if args.paths else DEFAULT_PATHS
    report = scan_paths(paths)
    print(json.dumps(report, indent=2, sort_keys=True), end="\n")
    return 0 if report["valid"] is True else 1


def _iter_checked_files(paths: Sequence[Path]) -> Iterator[Path]:
    for path in sorted((item.resolve() for item in paths), key=str):
        if path.is_file():
            if _should_scan_file(path):
                yield path
            continue
        if path.is_dir():
            for candidate in sorted(path.rglob("*"), key=str):
                if candidate.is_file() and _should_scan_file(candidate):
                    yield candidate


def _should_scan_file(path: Path) -> bool:
    if path.suffix not in CHECKED_SUFFIXES:
        return False
    return not _is_skipped_path(path)


def _is_skipped_path(path: Path) -> bool:
    return any(part in SKIPPED_PARTS or part.endswith(".egg-info") for part in path.parts)


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
