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

ACTIVATE_TF = "intel_tf/bin/activate"
ACTIVATE_PYT = "intel_pyt/bin/activate"
ACTIVATE_TEST = "test_env/bin/activate"
ACTIVATE_DOCS = $(ACTIVATE_TEST)
ACTIVATE_NOTEBOOK = $(ACTIVATE_TEST)

LISTEN_IP ?= 127.0.0.1
LISTEN_PORT ?= 9090
DOCS_DIR ?= ../docs

venv-test:
	@echo "Creating a virtualenv test_env..."
	@test -d test_env || virtualenv -p python test_env

	@echo "Building the XAI API in test_env env..."
	@. $(ACTIVATE_TEST) && pip install --editable .[test]

# TODO: running all tests in one pytest session randomly causes torch test to hang at last epoch
test-torch: venv-test
	@echo "Testing the API..."
	@. $(ACTIVATE_TEST) && PYTHONPATH="$(CURDIR)/model_card_gen/tests" pytest -s model_card_gen/tests/test_end_to_end_torch.py

test-mcg: test-torch
	@echo "Testing the API..."
	@. $(ACTIVATE_TEST) && PYTHONPATH="$(CURDIR)/model_card_gen/tests" pytest -s -k "not torch"

install:
	@pip install --editable .

xai-whl:
	@python setup.py bdist_wheel

clean:
	@rm -rf build dist intel_xai_tools.egg-info
	@rm -rf test_env

test-explainer: venv-test 
	@. $(ACTIVATE_TEST) && pytest explainer/tests

test: clean test-mcg test-explainer

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
	bash run_notebooks.sh $(CURDIR)/../notebooks/explainer/imagenet_with_cam/ExplainingImageClassification.ipynb

dist: venv-test
	@echo "Create binary wheel..."
	@. $(ACTIVATE_DOCS) && python setup.py bdist_wheel

check-dist: dist
	@echo "Testing the wheel..."
	@. $(ACTIVATE_DOCS) && \
	pip install twine && \
	python setup.py bdist_wheel && \
	twine check dist/*
