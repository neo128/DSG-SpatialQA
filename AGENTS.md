# DSG-SpatialQA Agent Guide

## Project Conventions
- Keep changes minimal and scoped to the requested DSG-SpatialQA Lab MVP.
- Prefer the repository's existing stack. If none exists, use a small Python package.
- Runtime behavior must be deterministic: no real network calls, no current-time reads, and no random output.
- External AI, robot, simulator, and service integrations must be mocked or omitted.
- Timestamps and steps must be supplied explicitly by callers.

## Verification Commands
- Install: `python -m pip install -e ".[dev]"`
- Lint: `python -m ruff check .`
- Typecheck: `python -m mypy src tests`
- Test: `python -m pytest -q`
- Build: `python -m build`

## Minimal Change Principle
- Do not make unrelated refactors or formatting-only changes.
- Add only files and dependencies required to verify the MVP.
- Keep in-memory data structures simple until a real integration requires more.
