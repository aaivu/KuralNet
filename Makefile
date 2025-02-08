.PHONY: venv setup sync test lint format-check format add remove clear

venv:
	poetry shell

setup:
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
