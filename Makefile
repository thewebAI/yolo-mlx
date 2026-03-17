# Copyright (c) 2026 webAI, Inc.

PYTHON ?= python3

.PHONY: install install-dev lint format test check pre-commit-install pre-commit-run

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e .[dev]

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

format:
	$(PYTHON) -m ruff check --fix src tests
	$(PYTHON) -m ruff format src tests

test:
	$(PYTHON) -m pytest

check: lint test

pre-commit-install:
	$(PYTHON) -m pre_commit install

pre-commit-run:
	$(PYTHON) -m pre_commit run --all-files
