# Contributing to YOLO26 MLX

Thanks for contributing. This project welcomes fixes, features, and docs improvements.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

For weight conversion work, also install conversion dependencies:

```bash
pip install -e .[convert]
```

## Common commands

```bash
make format
make lint
make test
make check
```

## Pre-commit hooks

Install hooks once:

```bash
make pre-commit-install
```

Run on all files:

```bash
make pre-commit-run
```

## Pull request checklist

- Keep changes focused and well-scoped.
- Add or update tests when behavior changes.
- Run `make check` before opening a PR.
- Update docs when CLI or workflow changes.

## CI expectations

CI runs lint + tests for Python 3.10/3.11/3.12 on every PR.
PRs should be green before review/merge.
