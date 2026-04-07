# Repository Guidelines

## Project Structure & Module Organization
`src/lottery/` contains the application package. Use `cli.py` and `__main__.py` for command entry points, then keep domain logic in the matching subpackage: `fetcher/`, `store/`, `analysis/`, `features/`, `models/`, `ensemble/`, `filters/`, and `visualization/`. Runtime settings live in `config.yaml`. Generated artifacts belong in `data/` and `output/`; `data/lottery.db`, imported CSVs, charts, and model files are ignored and should not be committed.

`tests/` mirrors the package by domain (`tests/test_analysis/`, `tests/test_models/`, etc.). Add concrete `test_*.py` files inside the relevant folder instead of mixing unrelated scenarios.

## Build, Test, and Development Commands
Create an environment with `python -m venv .venv` and activate it with `.venv\Scripts\activate` on Windows. Install the project in editable mode with `pip install -e .[dev]`; add `.[ml]` or `.[dl]` when working on XGBoost or LSTM code paths.

Useful commands:
- `python -m lottery --help`: show the CLI surface without relying on a global script.
- `lottery update`: fetch the latest draw data into SQLite.
- `lottery analyze --recent 200 --show-charts`: run analysis and save plots to `output/charts/`.
- `lottery predict --ensemble`: run the default ensemble prediction flow.
- `pytest` or `pytest --cov=lottery`: run tests and optional coverage.
- `ruff check src tests` and `mypy src`: run static checks defined in `pyproject.toml`.

## Coding Style & Naming Conventions
Target Python 3.10+ and use 4-space indentation. Ruff enforces a 100-character line length. Follow the existing style: `from __future__ import annotations`, explicit type hints, small focused functions, and concise docstrings. Use `snake_case` for modules, functions, and variables, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants.

## Testing Guidelines
Use `pytest` for all new coverage. Name test files `test_<behavior>.py` and test functions `test_<expected_result>`. Prefer deterministic fixtures and CSV-based inputs; do not rely on live network access in tests. When touching fetch, store, or model orchestration logic, add or extend tests in the matching `tests/test_<area>/` package.

## Commit & Pull Request Guidelines
Current history uses short imperative subjects (`Initial commit`), so keep commit titles brief and action-oriented, for example `Add CSV import validation`. Pull requests should describe the user-visible change, list the verification commands you ran, and call out any edits to `config.yaml`, dependencies, or generated outputs. Include screenshots only when chart rendering or CLI presentation changes.
