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
from explainer.version import __version__
import explainer
os.environ['KMP_WARNINGS'] = 'off'
project = 'Intel® Explainable AI Tools'
add_module_names = False
author = 'IntelAI'
bibtex_bibfiles = ['references.bib']
comments_config = {'hypothesis': False, 'utterances': False}
copyright = '2022'
exclude_patterns = [
  'requirements/**',
  'usecases/**',
  'xai/**',
  'explainer/criteria.md',
  'explainer/design.md',
  'explainer/examples/*.ipynb',
  'explainer/examples/model_layers.md',
  'explainer/examples/partitionexplainer.md',
  'explainer/examples/ExplainingDeepLearningModels.md',
  'explainer/examples/TorchVision_CIFAR_Interpret.ipynb',
  'explainer/examples/Explaining_Transformers.ipynb',
  'explainer/examples/heart_disease.ipynb',
  'explainer/examples/house_prices_regression.ipynb',
  'explainer/examples/multiclass_classification.ipynb',
  'explainer/examples/test_explainer.ipynb',
  'explainer/examples/zero_shot_learning.ipynb',
  'explainer/examples/vit_transformer.ipynb',
  'explainer/experiments/*.ipynb',
  '**.ipynb_checkpoints', 
  '.DS_Store',  
  'Thumbs.db', 
  '_build',
  'conf.py'
]
nb_execution_allow_errors = True
nb_execution_excludepatterns = []
nb_execution_in_temp = False
nb_execution_timeout = 600
nb_execution_mode = "cache"
nb_output_stderr = 'show'

release = __version__
version = __version__
extensions = [
  'myst_parser',
  'sphinx_togglebutton',
  'sphinx_copybutton',
  'sphinx_comments',
  'sphinx_design',
  'sphinx_exec_code',
  'sphinx_external_toc',
  'sphinx.ext.intersphinx',
  'sphinx.ext.autodoc',
  'sphinx.ext.napoleon',
  'sphinx.ext.viewcode',
  'sphinxcontrib.mermaid',
  'sphinxcontrib.bibtex',
]
external_toc_path = "toc.yml"
external_toc_exclude_missing = False

html_baseurl = ''
html_favicon = ''
html_js_files = ['https://cdnjs.cloudflare.com/ajax/libs/mermaid/8.9.1/mermaid.js']
html_sourcelink_suffix = ''
html_theme = 'sphinx_rtd_theme'
html_theme_options = {'logo_only': False}
html_title = 'Explainer'
myst_heading_anchors = 4
intersphinx_mapping = {
  'ebp': ['https://executablebooks.org/en/latest/', None],
  'myst-parser': ['https://myst-parser.readthedocs.io/en/latest/', None],
  'sphinx': ['https://www.sphinx-doc.org/en/master', None],
  'nbformat': ['https://nbformat.readthedocs.io/en/latest', None]
}
language = 'en'
latex_engine = 'pdflatex'
myst_enable_extensions = [
  'deflist', 
  'colon_fence', 
  'linkify', 
  'substitution', 
  'tasklist',
  'dollarmath',
  'html_image'
]
myst_url_schemes = ['mailto', 'http', 'https']
numfig = True
# pygments_style = 'sphinx'
suppress_warnings = [
  'myst.domains',
  'mystnb.unknown_mime_type',
  'myst.nested_header'
]
use_multitoc_numbering = True
html_css_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css"
]
