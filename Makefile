PYTHON ?= python3
VENV   := .venv
BIN    := $(VENV)/bin
LGA    ?= Willoughby
STATE  ?= NSW
CLI    := $(BIN)/property-orientation --lga "$(LGA)" --state $(STATE)

.PHONY: help setup data run figures webmap all lookup test lint format notebook clean

help:  ## Show this help
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-9s %s\n", $$1, $$2}'

setup:  ## Create .venv and install the package with all extras (PYTHON=python3.12 to choose)
	@$(PYTHON) -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else "Python 3.10+ is needed: run make setup PYTHON=python3.12 (or another 3.10+ interpreter)")'
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[viz,dev]"

data:  ## Download G-NAF, OpenStreetMap and NSW cadastre data into data/raw/
	$(CLI) fetch

run:  ## Compute orientation + validation and write results/
	$(CLI) run

figures:  ## Render README figures into docs/images/ (Willoughby only)
	$(CLI) figures

webmap:  ## Build the interactive map in docs/map/
	$(CLI) webmap

all: data run figures webmap  ## Run the whole pipeline end to end

lookup:  ## Which way does an address face? make lookup ADDRESS="22 The Parapet, Castlecrag"
	@test -n "$(ADDRESS)" || { echo 'Usage: make lookup ADDRESS="22 The Parapet, Castlecrag"'; exit 2; }
	@$(CLI) lookup "$(ADDRESS)"

test:  ## Run the test suite (offline)
	$(BIN)/pytest

lint:  ## Lint and check formatting
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format:  ## Auto-format and fix lint
	$(BIN)/ruff format .
	$(BIN)/ruff check --fix .

notebook:  ## Re-execute the walkthrough notebook in place
	$(BIN)/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.record_timing=False notebooks/walkthrough.ipynb

clean:  ## Remove downloaded and intermediate data (keeps results/)
	rm -rf data/raw data/processed .pytest_cache .ruff_cache
