.PHONY: venv setup sync test lint format-check format add remove

venv:
	poetry shell

setup:
	poetry install && poetry run pre-commit install --hook-type pre-commit --hook-type pre-push

sync:
	poetry install

test:
	poetry run pytest

lint:
	poetry run flake8

format-check:
	poetry run black --check . && poetry run isort --check .

format:
	poetry run black . && poetry run isort .


clear:
	@echo "Cleaning up..."
	rm -rf __pycache__ .pytest_cache poetry.lock
	@echo "Done."