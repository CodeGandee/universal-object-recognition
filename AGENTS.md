# Repository Guidelines

## Project Structure & Module Organization
- Code: `src/univ_obj/` (package scaffold; implementation pending)
- Tests: `tests/` (placeholder; add `test_*.py` here)
- Docs: `docs/` (Chinese design notes: overview, architecture, plan)
- Tooling: `pyproject.toml` (hatchling, deps), `.pixi/` + `pixi.lock` (env), `setup-envs.sh` (proxy helper)
- Misc: `magic-context/` (supporting materials; not loaded at runtime)

## Build, Test, and Development Commands
- Environment (Pixi): `pixi install` then `pixi shell` (Python 3.12)
- Editable install: `python -m pip install -e .`
- Lint: `ruff check .`  |  Format: `ruff format .`
- Type-check: `mypy src`
- Tests (when added): `pytest -q` (recommend adding `pytest` as a dev dep)

## Coding Style & Naming Conventions
- Python 3.12+, 4-space indentation, UTF‑8.
- Use type hints; keep `mypy` clean for `src/`.
- Names: modules/packages `lower_snake_case`; classes `PascalCase`; functions/vars `snake_case`; constants `UPPER_SNAKE_CASE`.
- Prefer small, pure functions; avoid hidden global state.
- Run `ruff format` before committing; keep `ruff check` warnings at zero.

## Testing Guidelines
- Framework: `pytest` with plain asserts; place files as `tests/test_*.py`.
- Scope: unit tests for pure logic; lightweight integration tests for pipeline boundaries when implemented.
- Targets: aim for ≥80% line coverage (future CI may enforce).
- Keep tests hermetic (no network/GPU); mock detectors/VLMs and file I/O.

## Commit & Pull Request Guidelines
- Style: Conventional Commits (e.g., `feat(pipeline): rank main object via VLM`).
- Keep commits focused and descriptive; English preferred (Chinese notes welcome).
- PRs: include clear summary, linked issues, rationale, and updated docs/tests; add screenshots/logs for UI or CLI changes.
- Ensure `ruff`, `mypy`, and tests pass locally before requesting review.

## Security & Configuration Tips
- Do not commit credentials, model weights, or large binaries; use `.gitignore` and external storage.
- For network-restricted setups, run `./setup-envs.sh --proxy auto` before dependency/model downloads.
- Record any required environment variables in `docs/使用指南.md` when features land.
