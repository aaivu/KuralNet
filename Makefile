.PHONY: venv setup sync test lint format-check format add remove clear build help

help:
	@echo "Available targets:"
	@echo "  venv           - Create and activate a virtual environment"
	@echo "  setup          - Install pip and project dependencies"
	@echo "  sync           - Install project dependencies"
	@echo "  test           - Run tests using pytest"
	@echo "  lint           - Run flake8 for linting"
	@echo "  format-check   - Check code formatting using black and isort"
	@echo "  format         - Format code using black and isort"
	@echo "  build          - Build the package"
	@echo "  add            - Add a new dependency to requirements.txt"
	@echo "  remove         - Remove a dependency from requirements.txt"
	@echo "  clear          - Clean up temporary files like __pycache__ and .pytest_cache"

venv:
	python3 -m venv .venv
	@echo "Virtual environment created. Run 'source .venv/bin/activate' to activate it."

setup: venv
	. .venv/bin/activate && pip install --upgrade pip
	. .venv/bin/activate && pip install -r requirements.txt
	. .venv/bin/activate && pip install -r dev-requirements.txt
	@echo "Dependencies installed."

sync:
	. .venv/bin/activate && pip install -r requirements.txt
	. .venv/bin/activate && pip install -r dev-requirements.txt
	@echo "Dependencies synchronized."

test:
	. .venv/bin/activate && pytest

lint:
	. .venv/bin/activate && flake8 kuralnet/ data/

format-check:
	. .venv/bin/activate && black --check --line-length 79 kuralnet/ data/ && isort --check kuralnet/ data/

format:
	. .venv/bin/activate && black --line-length 79 kuralnet/ && isort kuralnet/

test:
	. .venv/bin/activate && pytest .

build:
	python setup.py sdist bdist_wheel
	@echo "Package built."

add:
	@read -p "Enter the package name to add: " package; \
	. .venv/bin/activate && pip install $$package && \
	echo $$package >> requirements.txt && \
	echo "$$package added to requirements.txt."

add-dev:
	@read -p "Enter the package name to add: " package; \
	. .venv/bin/activate && pip install $$package && \
	echo $$package >> dev-requirements.txt && \
	echo "$$package added to dev-requirements.txt."

remove:
	@read -p "Enter the package name to remove: " package; \
	sed -i '/$$package/d' requirements.txt && \
	echo "$$package removed from requirements.txt."

remove-dev:
	@read -p "Enter the package name to remove: " package; \
	sed -i '/$$package/d' dev-requirements.txt && \
	echo "$$package removed from dev-requirements.txt."

download_dataset:
	. .venv/bin/activate && python -m meta.download_datasets

meta_extract:
	. .venv/bin/activate && python -m meta.meta_extractor

train:
	. .venv/bin/activate && python -m scripts.run_training --config configs/train_config.yaml

clear:
	@echo "Cleaning up..."
	rm -rf __pycache__ .pytest_cache dist build *.egg-info
	@echo "Done."
