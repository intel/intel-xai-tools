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
ACTIVATE_TEST = "$(VENV_DIR)/bin/activate"
ACTIVATE_LINT = "$(VENV_LINT)/bin/activate"
ACTIVATE_DOCS = $(ACTIVATE_TEST)
ACTIVATE_NOTEBOOK = $(ACTIVATE_TEST)

LISTEN_IP ?= 127.0.0.1
LISTEN_PORT ?= 9090
DOCS_DIR ?= docs

venv-test: poetry-lock
	@echo "Creating a virtualenv $(VENV_DIR)..."
	@poetry install --with test --extras all
	@poetry run python -m pip install --no-cache-dir --no-deps \
		asttokens==2.4.1 \
		executing==2.0.1 \
		ipython==8.10 \
		jupyter-server==2.14.1 \
		pure-eval==0.2.2 \
		stack-data==0.6.3

venv-lint: 
	@echo "Creating a virtual environment for linting $(VENV_LINT)..."
	@test -d $(VENV_LINT) || python -m virtualenv $(VENV_LINT) || python3 -m virtualenv $(VENV_LINT)
	@echo "Installing lint dependencies..."
	@. $(ACTIVATE_LINT) && pip install --no-cache-dir --no-deps \
		flake8==7.0.0 \
		black==24.4.2

test-mcg: venv-test
	@echo "Testing the Model Card Gen API..."
	@. $(ACTIVATE_TEST) && pytest model_card_gen/tests

install: poetry-lock
	@poetry install --extras all

build-whl:
	@poetry build

clean:
	@rm -rf build dist intel_ai_safety.egg-info
	@rm -rf $(VENV_DIR)

test-explainer: venv-test
	@echo "Testing the Explainer API..."
	@. $(ACTIVATE_TEST) && pytest plugins/explainers/attributions/tests
	@. $(ACTIVATE_TEST) && pytest plugins/explainers/attributions-hugging-face/tests
	@. $(ACTIVATE_TEST) && pytest plugins/explainers/cam-tensorflow/tests
	@. $(ACTIVATE_TEST) && pytest plugins/explainers/cam-pytorch/tests
	@. $(ACTIVATE_TEST) && pytest plugins/explainers/metrics/tests

test: test-mcg test-explainer

venv-docs: venv-test ${DOCS_DIR}/requirements-docs.txt
	@echo "Installing docs dependencies..."
	@. $(ACTIVATE_DOCS) && pip install -r ${DOCS_DIR}/requirements-docs.txt

html: venv-docs
	@echo "Building Sphinx documentation..."
	@. $(ACTIVATE_DOCS) && $(MAKE) -C ${DOCS_DIR} clean html

test-docs: html
	@echo "Testing Sphinx documentation..."
	@. $(ACTIVATE_DOCS) && $(MAKE) -C ${DOCS_DIR} doctest

test-notebook: venv-test
	@echo "Testing Jupyter notebooks..."
	@. $(ACTIVATE_NOTEBOOK) && \
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
	@poetry update --lock
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
