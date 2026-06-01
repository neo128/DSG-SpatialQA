from __future__ import annotations

import importlib.util
import sys
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast


ROOT = Path(__file__).resolve().parents[1]
VERIFY_SCRIPT = ROOT / "scripts" / "verify.py"


class CheckProtocol(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def argv(self) -> tuple[str, ...]: ...


class VerificationChecksFn(Protocol):
    def __call__(self, *, include_install: bool = True) -> tuple[CheckProtocol, ...]: ...


CommandRunner = Callable[[tuple[str, ...], Path], int]


class RunVerificationFn(Protocol):
    def __call__(
        self,
        checks: Sequence[CheckProtocol],
        *,
        cwd: Path,
        runner: CommandRunner,
    ) -> int: ...


@dataclass(frozen=True)
class FakeCheck:
    name: str
    argv: tuple[str, ...]


def load_verify_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verify_script", VERIFY_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_verification_checks_run_required_gates_in_order() -> None:
    module = load_verify_script()
    verification_checks = cast(
        VerificationChecksFn,
        getattr(module, "verification_checks"),
    )

    checks = verification_checks()

    assert [check.name for check in checks] == [
        "install",
        "lint",
        "typecheck",
        "determinism",
        "test",
        "build",
        "evaluation_suite",
    ]
    assert checks[0].argv == (sys.executable, "-m", "pip", "install", "-e", ".[dev]")
    assert checks[1].argv == (sys.executable, "-m", "ruff", "check", ".")
    assert checks[2].argv == (sys.executable, "-m", "mypy", "src", "tests", "scripts")
    assert checks[3].argv == (sys.executable, "scripts/check_determinism.py")
    assert checks[4].argv == (sys.executable, "-m", "pytest", "-q")
    assert checks[5].argv == (sys.executable, "-m", "build")
    assert checks[6].argv[:2] == (sys.executable, "-c")
    assert checks[6].argv[2].startswith("from dsg_spatialqa_lab")


def test_verification_checks_can_skip_editable_install() -> None:
    module = load_verify_script()
    verification_checks = cast(
        VerificationChecksFn,
        getattr(module, "verification_checks"),
    )

    checks = verification_checks(include_install=False)

    assert [check.name for check in checks] == [
        "lint",
        "typecheck",
        "determinism",
        "test",
        "build",
        "evaluation_suite",
    ]


def test_run_verification_stops_on_first_failed_check(tmp_path: Path) -> None:
    module = load_verify_script()
    run_verification = cast(
        RunVerificationFn,
        getattr(module, "run_verification"),
    )
    checks = (
        FakeCheck("first", ("first",)),
        FakeCheck("second", ("second",)),
        FakeCheck("third", ("third",)),
    )
    seen: list[str] = []

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        assert cwd == tmp_path
        seen.append(argv[0])
        return 7 if argv[0] == "second" else 0

    assert run_verification(checks, cwd=tmp_path, runner=runner) == 7
    assert seen == ["first", "second"]


def test_package_declares_pep561_type_marker() -> None:
    marker = ROOT / "src" / "dsg_spatialqa_lab" / "py.typed"
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert marker.read_text(encoding="utf-8").strip() == ""
    assert pyproject["tool"]["setuptools"]["package-data"] == {
        "dsg_spatialqa_lab": ["py.typed"]
    }
