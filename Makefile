.PHONY: venv setup sync test lint format-check format add remove clear help

help:
	@echo "Available targets:"
	@echo "  venv           - Activate the poetry virtual environment"
	@echo "  setup          - Install poetry and dependencies"
	@echo "  sync           - Install project dependencies"
	@echo "  test           - Run tests using pytest"
	@echo "  lint           - Run flake8 for linting"
	@echo "  format-check   - Check code formatting using black and isort"
	@echo "  format         - Format code using black and isort"
	@echo "  clear          - Clean up temporary files like __pycache__ and .pytest_cache"

venv:
	poetry shell

setup:
	pip install poetry
	poetry install

sync:
	poetry install

test:
	poetry run pytest

lint:
	poetry run flake8 py/ src/

format-check:
	poetry run black --check py/ src/ && poetry run isort --check py/ src/

format:
	poetry run black py/ src/ && poetry run isort py/ src/

clear:
	@echo "Cleaning up..."
	rm -rf __pycache__ .pytest_cache poetry.lock
	@echo "Done."