.PHONY: venv setup sync test lint format-check format add remove clear build help

# Detect OS and set ACTIVATE accordingly.
ifeq ($(OS),Windows_NT)
    # For Windows using a Unix-like shell (e.g., Git Bash), you might still use:
    ACTIVATE = . .venv/Scripts/activate
    # If you're using CMD, consider using "call" (and note that Make might require additional tweaks):
    # ACTIVATE = call .venv\Scripts\activate
else
    ACTIVATE = . .venv/bin/activate
endif

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
	@echo "Virtual environment created. Activate it with:"
	@echo "    $(ACTIVATE)"

setup: venv
	$(ACTIVATE) && pip install --upgrade pip
	$(ACTIVATE) && pip install -r requirements.txt
	$(ACTIVATE) && pip install -r dev-requirements.txt
	@echo "Dependencies installed."

sync:
	$(ACTIVATE) && pip install -r requirements.txt
	$(ACTIVATE) && pip install -r dev-requirements.txt
	@echo "Dependencies synchronized."

test:
	$(ACTIVATE) && pytest

lint:
	$(ACTIVATE) && flake8 multilingual_speech_emotion_recognition/ data/

format-check:
	$(ACTIVATE) && black --check --line-length 79 multilingual_speech_emotion_recognition/ data/ && isort --check multilingual_speech_emotion_recognition/ data/

format:
	$(ACTIVATE) && black --line-length 79 multilingual_speech_emotion_recognition/ data/ && isort multilingual_speech_emotion_recognition/ data/

build:
	python setup.py sdist bdist_wheel
	@echo "Package built."

add:
	@read -p "Enter the package name to add: " package; \
	$(ACTIVATE) && pip install $$package && \
	echo $$package >> requirements.txt && \
	echo "$$package added to requirements.txt."

add-dev:
	@read -p "Enter the package name to add: " package; \
	$(ACTIVATE) && pip install $$package && \
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

clear:
	@echo "Cleaning up..."
	rm -rf __pycache__ .pytest_cache dist build *.egg-info
	@echo "Done."
