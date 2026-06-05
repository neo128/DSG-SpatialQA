from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dsg_spatialqa_lab.navigation.trajectory_audit import (
    compare_trajectory_protocols,
    comparison_markdown,
    save_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Compare fixed, diagnostic, and reachable NBV trajectory audits.",
    )
    parser.add_argument("--fixed-audit", type=Path, required=True)
    parser.add_argument("--diagnostic-audit", type=Path, required=True)
    parser.add_argument("--nbv-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        report = compare_trajectory_protocols(
            _load_json(args.fixed_audit),
            _load_json(args.diagnostic_audit),
            _load_json(args.nbv_audit),
        )
        save_json(report, args.output)
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(
            comparison_markdown(report),
            encoding="utf-8",
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _emit({"valid": False, "error": str(exc)})
        return 1
    _emit({"valid": True, "path": str(args.output), "markdown_path": str(args.markdown_output), "judgement": report["judgement"]})
    return 0


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True) + "\n", end="")


if __name__ == "__main__":
    raise SystemExit(main())

