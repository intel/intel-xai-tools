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

# Note: These are just placeholders for future additions to Makefile.
# You can remove these comments later.
ACTIVATE_MCG = "mcg_env/bin/activate"
ACTIVATE_TF = "intel_tf/bin/activate"
ACTIVATE_PYT = "intel_pyt/bin/activate"
ACTIVATE_TEST = "mcg_tests/bin/activate"
ACTIVATE_DOCS = $(ACTIVATE_TEST)

venv_test:
	@echo "Creating a virtualenv mcg_tests..."
	@test -d mcg_tests || virtualenv -p python mcg_tests

	@echo "Building the TLT API in mcg_tests env..."
	@. $(ACTIVATE_TEST) && pip install --editable .[test]

# TODO: running all tests in one py.test session randomly causes torch test to hang at last epoch
test-torch: venv_test
	@echo "Testing the API..."
	@. $(ACTIVATE_TEST) && PYTHONPATH="$(CURDIR)/model_card_gen/tests" py.test -s model_card_gen/tests/test_end_to_end_torch.py

test: test-torch
	@echo "Testing the API..."
	@. $(ACTIVATE_TEST) && PYTHONPATH="$(CURDIR)/model_card_gen/tests" py.test -s -k "not torch"


venv_docs: venv_test $(CURDIR)/docs/requirements-docs.txt
	@echo "Installing docs dependencies..."
	@. $(ACTIVATE_DOCS) && pip install -r $(CURDIR)/docs/requirements-docs.txt

html: venv_docs
	@echo "Building Sphinx documentation..."
	# @. $(ACTIVATE_DOCS) && $(MAKE) -C docs clean html

test_docs: html
	@echo "Testing Sphinx documentation..."
	# @. $(ACTIVATE_DOCS) && $(MAKE) -C docs doctest

dist: venv_docs
	@echo "Create binary wheel..."
	# @. $(ACTIVATE_DOCS) && python setup.py bdist_wheel

check_dist: dist
	@echo "Testing the wheel..."
	# @. $(ACTIVATE_DOCS) && \
	# pip install twine && \
	# python setup.py bdist_wheel && \
	# twine check dist/*

clean:
	rm -rf mcg_tests
