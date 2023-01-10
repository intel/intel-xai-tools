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

all: html

html:: clean
	sphinx-build --keep-going . ./_build/html -b html

clean::
	rm -rf _build

publish-install::
	pip install ghp-import~=2.1.0

publish::
	ghp-import -n -p -f _build/html

LISTEN_IP ?= 127.0.0.1
LISTEN_PORT ?= 9999
serve::
	python -m http.server --directory ./_build/html ${LISTEN_PORT} --bind ${LISTEN_IP}

install::
	pip install Sphinx==5.3.0 && pip install -U -r ./requirements.txt

autodoc::
	sphinx-apidoc -o /tmp explainer
