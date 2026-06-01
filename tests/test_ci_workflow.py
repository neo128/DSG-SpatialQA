from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "verify.yml"


def test_ci_workflow_reuses_local_verification_entrypoint() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "name: Verify" in workflow
    assert "python-version: \"3.11\"" in workflow
    assert "python scripts/verify.py" in workflow
    assert "python -m ruff check ." not in workflow
    assert "python -m mypy src tests" not in workflow
    assert "python -m pytest -q" not in workflow
    assert "python -m build" not in workflow


def test_ci_workflow_runs_on_push_and_pull_request() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "push:" in workflow
    assert "pull_request:" in workflow
    assert "branches: [\"**\"]" in workflow
