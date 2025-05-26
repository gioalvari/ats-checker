.PHONY: setup lint test

setup:
	uv venv
	. .venv/bin/activate && uv pip install -r requirements.txt && pre-commit install

lint:
	. .venv/bin/activate && pre-commit run --all-files

test:
	. .venv/bin/activate && pytest
