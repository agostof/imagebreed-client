# Test layout, marker contract, and CI suites

This project uses `pytest` with explicit markers and strict marker validation.
The canonical marker/pytest options live in `pytest.ini`.

## Directory structure

- `tests/conftest.py`
  - Shared fixtures and default marker assignment for deterministic CI selection.
- `tests/helpers/`
  - Shared test-only helper assertions and utilities.
- `tests/unit/`
  - Fast isolated unit tests.
- `tests/integration/`
  - Cross-module and IO-boundary integration tests.
- `tests/regression/`
  - Regression tests that preserve previously-fixed behavior.

## Marker meanings

Declared markers (see `pytest.ini`):

- `unit`: fast isolated tests; default marker for tests outside `integration/` and `regression/`.
- `smoke`: high-signal sanity checks that are quick and stable.
- `integration`: cross-module/IO tests.
- `slow`: expensive tests intended for scheduled/manual pipelines.
- `legacy_geojson`: compatibility coverage for legacy/non-PixelGeoJSON inputs.
- `regression`: behavior lock-in for prior defects.
- `cli`: command-line invocation/argument handling tests.

## Blocking vs non-blocking suites

Workflow file: `.github/workflows/ci.yml`

- **Blocking (PR + main push):** `PR Fast Suite`
  - Marker expression: `unit or smoke`
  - Purpose: required fast feedback baseline.
- **Non-blocking (scheduled/manual):** `Extended Integration Suite`
  - Marker expression: `integration or slow`
  - Purpose: broader quality coverage without slowing PR iteration.
- **Guard job (all triggers):** `Marker Guard`
  - Fails if unknown custom pytest markers appear.

## Local commands

Install baseline test dependencies:

```bash
pip install -r requirements-dev.txt
```

Run the same commands used in CI via the shared script:

```bash
# PR-equivalent fast suite
scripts/ci/test_matrix.sh pr

# Scheduled/manual extended suite
scripts/ci/test_matrix.sh extended

# Marker regression guard
scripts/ci/test_matrix.sh markers
```

You can also run equivalent raw pytest expressions directly:

```bash
pytest -m "unit or smoke"
pytest -m "integration or slow"
```
