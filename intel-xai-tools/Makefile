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

DOCS_DIR ?= ../docs
explainer::
	pip install -e .

explainer-whl::
	python setup.py bdist_wheel

explainer-install::
	pip install dist/intel_xai_tools-0.2.0-py2.py3-none-any.whl

clean::
	@rm -rf build dist intel_xai_tools.egg-info

test::
	pytest tests

bump-version::
	pip install bump2version==1.0.1
	bump2version --current-version 0.2.0 minor explainer/version.py

docs-install::
	cd ${DOCS_DIR} && make install

docs::
	cd ${DOCS_DIR} && make html

docs-publish::
	ghp-import -n -p -f ${DOCS_DIR}/_build/html

LISTEN_IP ?= 127.0.0.1
LISTEN_PORT ?= 9090
docs-serve::
	@echo serve docs locally on ${LISTEN_IP}:${LISTEN_PORT}
	@python -m http.server --directory ${DOCS_DIR}/_build/html ${LISTEN_PORT} --bind ${LISTEN_IP}
