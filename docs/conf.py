#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import sys, os
from intel_ai_safety.explainer.version import __version__
from intel_ai_safety import explainer

project = 'Intel® Explainable AI Tools'
author = 'IntelAI'
copyright = '2023, Intel'
exclude_patterns = [
    '_build',
    'conf.py',
]

release = __version__
version = __version__
extensions = [
  'myst_parser',
  'nbsphinx',
  'nbsphinx_link',
  'sphinx_design',
  'sphinx_external_toc',
  'sphinx.ext.intersphinx',
  'sphinx.ext.autodoc',
  'sphinx.ext.napoleon',
  'sphinx.ext.viewcode',
  'sphinx.ext.doctest',
]
external_toc_path = "toc.yml"
external_toc_exclude_missing = False
html_theme = 'sphinx_rtd_theme'
nbsphinx_execute = 'never'
