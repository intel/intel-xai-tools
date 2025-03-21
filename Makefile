#
# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
#

VENV_DIR = ".venv"
VENV_LINT = ".venv/lint"
VENV_BENCHMARK = ".venv/benchmark"
VENV_EXPLAINER = ".venv/explainer"
VENV_MCG = ".venv/mcg"
VENV_DOCS = ".venv/docs"
VENV_NOTEBOOK = ".venv/notebooks"

ACTIVATE_TEST = "$(VENV_DIR)/bin/activate"
ACTIVATE_LINT = "$(VENV_LINT)/bin/activate"
ACTIVATE_BENCHMARK = "$(VENV_BENCHMARK)/bin/activate"
ACTIVATE_EXPLAINER = "$(VENV_EXPLAINER)/bin/activate"
ACTIVATE_MCG = "$(VENV_MCG)/bin/activate"
ACTIVATE_DOCS = "$(VENV_DOCS)/bin/activate"
ACTIVATE_NOTEBOOK = "$(VENV_NOTEBOOK)/bin/activate"


LISTEN_IP ?= 127.0.0.1
LISTEN_PORT ?= 9090
DOCS_DIR ?= docs

venv-test: poetry-lock
	@echo "Creating a virtualenv $(VENV_DIR)..."
	@poetry install --with test --extras all

venv-lint: 
	@echo "Creating a virtual environment for linting $(VENV_LINT)..."
	@test -d $(VENV_LINT) || python -m virtualenv $(VENV_LINT) || python3 -m virtualenv $(VENV_LINT)
	@. $(ACTIVATE_LINT) && pip install --no-cache-dir --no-deps \
		black==25.1.0 \
		flake8==7.1.2

venv-mcg:
	@echo "Setting up virtual environment and downloading dependencies for ModelCardGen $(VENV_MCG)..."
	@test -d $(VENV_MCG) || python -m virtualenv $(VENV_MCG) || python3 -m virtualenv $(VENV_MCG)
	@. $(ACTIVATE_MCG) && poetry install --with test --extras model-card

venv-benchmark:
	@echo "Creating a virtual environment and downloading dependencies for Benchmarking $(VENV_BENCHMARK)..."
	@test -d $(VENV_BENCHMARK) || python -m virtualenv $(VENV_BENCHMARK) || python3 -m virtualenv $(VENV_BENCHMARK)
	@. $(ACTIVATE_BENCHMARK) && poetry install --with test --extras benchmark

venv-explainer:
	@echo "Setting up virtual environment and downloading dependencies for Explainers $(VENV_EXPLAINER)..."
	@test -d $(VENV_EXPLAINER) || python -m virtualenv $(VENV_EXPLAINER) || python3 -m virtualenv $(VENV_EXPLAINER)
	@. $(ACTIVATE_EXPLAINER) && poetry install --with test --extras explainer-all

install: poetry-lock
	@poetry install --extras all

build-whl:
	@poetry build

clean:
	@rm -rf build dist intel_ai_safety.egg-info
	@rm -rf $(VENV_DIR)

test-mcg: venv-mcg
	@echo "Testing the Model Card Gen API..."
	@. $(ACTIVATE_MCG) && pytest model_card_gen/tests

test-explainer: venv-explainer
	@echo "Testing the Explainer API..."
	@. $(ACTIVATE_EXPLAINER) && pytest plugins/explainers/attributions/tests
	@. $(ACTIVATE_EXPLAINER) && pytest plugins/explainers/attributions-hugging-face/tests
	@. $(ACTIVATE_EXPLAINER) && pytest plugins/explainers/cam-tensorflow/tests
	@. $(ACTIVATE_EXPLAINER) && pytest plugins/explainers/cam-pytorch/tests
	@. $(ACTIVATE_EXPLAINER) && pytest plugins/explainers/metrics/tests

test-benchmark: venv-benchmark
	@echo "Testing Benchmarking..."
	@. $(ACTIVATE_BENCHMARK) && pytest plugins/benchmark/classification_metrics/classification_metrics/tests

test: test-mcg test-explainer test-benchmark

venv-docs: ${DOCS_DIR}/requirements-docs.txt
	@echo "Installing docs dependencies..."
	@test -d $(VENV_DOCS) || python -m virtualenv $(VENV_DOCS) || python3 -m virtualenv $(VENV_DOCS)
	@. $(ACTIVATE_DOCS) && poetry install --with test
	@. $(ACTIVATE_DOCS) && pip install -r ${DOCS_DIR}/requirements-docs.txt

html: venv-docs
	@echo "Building Sphinx documentation..."
	@. $(ACTIVATE_DOCS) && $(MAKE) -C ${DOCS_DIR} clean html

test-docs: html
	@echo "Testing Sphinx documentation..."
	@. $(ACTIVATE_DOCS) && $(MAKE) -C ${DOCS_DIR} doctest

test-notebook:
	@echo "Testing Jupyter notebooks..."
	@test -d $(VENV_NOTEBOOK) || python -m virtualenv $(VENV_NOTEBOOK) || python3 -m virtualenv $(VENV_NOTEBOOK)
	@. $(ACTIVATE_NOTEBOOK) && poetry install --extras explainer-all && \
		poetry run pip install --no-cache-dir jupyter && \
	bash run_notebooks.sh $(CURDIR)/notebooks/explainer/imagenet_with_cam/ExplainingImageClassification.ipynb

stylecheck: venv-lint
	@echo "Checking code style..."
	@. $(ACTIVATE_LINT) flake8 . --config=tox.ini && echo "Code style is compatible with PEP 8 guidelines" || echo "Code style check failed. Please fix the above code style errors."

fix-codestyle: venv-lint
	@echo "Fixing code style..."
	@. $(ACTIVATE_LINT) black . --check --config=pyproject.toml

dist: build-whl
	@echo "Create binary wheel..."

check-dist: dist
	@echo "Testing the wheel..."
	@. $(ACTIVATE_DOCS) && \
	pip install twine && \
	twine check dist/*

bump: venv-test
	@poetry self add poetry-bumpversion
	@poetry version patch --dry-run
	@(cd model_card_gen ; poetry version patch --dry-run)
	@(cd explainer; poetry version patch --dry-run)
	@echo -n "Are you sure you want to make above changes? [y/N] " && read ans && [ $${ans:-N} = y ]
	@poetry version patch
	@(cd model_card_gen ; poetry version patch)
	@(cd explainer; poetry version patch)

poetry-lock:
	@echo "Lock all project dependency versions"
	@cd explainer && poetry lock && cd -
	@cd model_card_gen && poetry lock && cd -
	@cd plugins/explainers/attributions-hugging-face && poetry lock && cd -
	@cd plugins/explainers/attributions && poetry lock && cd -
	@cd plugins/explainers/cam-pytorch && poetry lock && cd -
	@cd plugins/explainers/cam-tensorflow && poetry lock && cd -
	@cd plugins/explainers/captum && poetry lock && cd -
	@cd plugins/explainers/metrics && poetry lock && cd -

poetry-lock-update:
	@echo "Update and Lock all project dependency versions"
	@poetry update --lock
	@cd model_card_gen && poetry update --lock && cd -
	@cd plugins/explainers/attributions-hugging-face && poetry update --lock && cd -
	@cd plugins/explainers/attributions && poetry update --lock && cd -
	@cd plugins/explainers/cam-pytorch && poetry update --lock && cd -
	@cd plugins/explainers/cam-tensorflow && poetry update --lock && cd -
	@cd plugins/explainers/captum && poetry update --lock && cd -
	@cd plugins/explainers/metrics && poetry update --lock && cd -
