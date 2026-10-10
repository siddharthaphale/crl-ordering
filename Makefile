.DEFAULT_GOAL := help
.PHONY: help setup test gpu lint check
help:
	@echo "setup: locked installation and pinned vendor; check: lint + CPU; gpu: required CUDA tests"
setup:
	uv sync --locked --extra experiments --extra cuda
	uv run --no-sync python -m crlo.utils.ogbench_vendor --install
test:
	PYTHONUNBUFFERED=1 uv run --no-sync pytest tests/cpu -q
gpu:
	PYTHONUNBUFFERED=1 MUJOCO_GL=egl uv run --no-sync pytest tests/gpu -q
lint:
	uv run --no-sync ruff check src experiments tests
	uv run --no-sync ruff format --check src experiments tests
check: lint test
