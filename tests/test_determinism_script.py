from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, cast

from _pytest.capture import CaptureFixture


ROOT = Path(__file__).resolve().parents[1]
DETERMINISM_SCRIPT = ROOT / "scripts" / "check_determinism.py"


class MainFn(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


class ScanPathsFn(Protocol):
    def __call__(self, paths: Sequence[Path]) -> dict[str, Any]: ...


def load_determinism_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("determinism_script", DETERMINISM_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_determinism_scan_accepts_clean_explicit_source_file(tmp_path: Path) -> None:
    module = load_determinism_script()
    scan_paths = cast(ScanPathsFn, getattr(module, "scan_paths"))
    source_path = tmp_path / "clean.py"
    source_path.write_text("VALUE = 1\n", encoding="utf-8")

    assert scan_paths((source_path,)) == {"valid": True, "matches": []}


def test_determinism_scan_reports_blocked_source_token(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    module = load_determinism_script()
    main = cast(MainFn, getattr(module, "main"))
    blocked_term = "".join(("rand", "om"))
    source_path = tmp_path / "blocked.py"
    source_path.write_text(f"import {blocked_term}\n", encoding="utf-8")

    assert main([str(source_path)]) == 1

    report = json.loads(capsys.readouterr().out)
    assert report == {
        "valid": False,
        "matches": [
            {
                "path": str(source_path),
                "line": 1,
                "pattern": blocked_term,
                "reason": "rand_source",
            }
        ],
    }


def test_determinism_scan_skips_generated_package_metadata(tmp_path: Path) -> None:
    module = load_determinism_script()
    scan_paths = cast(ScanPathsFn, getattr(module, "scan_paths"))
    blocked_term = "".join(("rand", "om"))
    metadata_dir = tmp_path / "pkg.egg-info"
    metadata_dir.mkdir()
    generated_source = metadata_dir / "generated.py"
    generated_source.write_text(f"import {blocked_term}\n", encoding="utf-8")

    assert scan_paths((tmp_path,)) == {"valid": True, "matches": []}
