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
import explainer
sys.path.insert(0, os.path.abspath('../explainer/explainer/explainers/lm_layers_explainer'))
sys.path.insert(0, os.path.abspath('../explainer/explainer/explainers/feature_attributions_explainer'))
sys.path.insert(0, os.path.abspath('../explainer/explainer/explainers/metrics_explainer'))
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
nb_custom_formats = {
  ".py": ["jupytext.reads", {"fmt": "light"}]
}
release = explainer.__version__
version = explainer.__version__
extensions = [
  'myst_nb',
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
  'sphinx_click.ext',
  'sphinxcontrib.bibtex',
]
external_toc_path = "toc.yml"
external_toc_exclude_missing = False
source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
    '.py': 'myst-nb',
    '.myst': 'myst-nb',
}
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
  'myst-nb': ['https://myst-nb.readthedocs.io/en/latest/', None],
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
myst_substitutions = {
  'AutoClasses': '<a href="https://huggingface.co/docs/transformers/main/en/model_doc/auto#auto-classes" target="_blank">AutoClasses</a>', 
	'DICEExample': '<a href="https://matthewpburruss.com/post/yaml/" target="_blank">PyYaml</a>', 
  'EntryPointsForPlugins': '<a href="https://setuptools.pypa.io/en/latest/userguide/entry_point.html#entry-points-for-plugins" target="_blank">Entry Points for Plugins</a>', 
  'ExplainableTransformers': '<a href="https://github.com/wilsonjr/explainable-transformers" target="_blank">explainable-transformers</a>', 
  'HOLZINGER202128': '<a href="https://www.sciencedirect.com/science/article/pii/S1566253521000142" target="_blank">Towards multi-modal causability with Graph Neural Networks enabling information fusion for explainable AI</a> {cite}`ZHU202253`', 
  'PluginArchitectureinPython': '<a href="https://dev.to/charlesw001/plugin-architecture-in-python-jla" target="_blank">Plugin Architecture in Python</a>', 
  'Glue': '<a href="http://docs.glueviz.org/en/stable" target="_blank">Glue</a>', 
  'Captum': '<a href="https://captum.ai/" target="_blank">Captum</a>', 
  'MetaPathFinders': '<a href="https://python.plainenglish.io/metapathfinders-or-how-to-change-python-import-behavior-a1cf3b5a13ec" target="_blank">How to Change Python Import Behavior with MetaPathFinders</a>', 
  'PathExplain': '<a href="https://github.com/suinleelab/path_explain" target="_blank">path-explain</a>', 
  'PEP451': '<a href="https://peps.python.org/pep-0451/" target="_blank">PEP451</i></a>', 
  'PEP690': '<a href="https://peps.python.org/pep-0690/" target="_blank">PEP690</i></a>', 
  'PythonInteroperabilitySpecifications': '<a href="https://packaging.python.org/en/latest/specifications/" target="_blank">python interoperability specifications</a>',
  'PythonEntryPointsDataModel': '<a href="https://packaging.python.org/en/latest/specifications/entry-points/#data-model" target="_blank">Data Model</a>',
  'PythonEntryPointsFunction': '<a href="https://docs.python.org/3/library/importlib.metadata.html#entry-points" target="_blank">entry points</a>',
  'PythonEntryPointsSpecification': '<a href="https://packaging.python.org/en/latest/specifications/entry-points/" target="_blank">python entry points specification</a>',
  'PyTestPlugins': '<a href="https://www.wheelodex.org/entry-points/pytest11/" target="_blank">plugins</a>', 
  'PyYaml': '<a href="https://matthewpburruss.com/post/yaml/" target="_blank">PyYaml</a>', 
  'Transformers': '<a href="https://huggingface.co/docs/transformers/index" target="_blank">Transformers</a>', 
  'TransformersInterpret': '<a href="https://github.com/cdpierse/transformers-interpret#install" target="_blank">transformers-interpret</a>', 
  'XAI': '<a href="https://en.wikipedia.org/wiki/Explainable_artificial_intelligence" target="_blank">Explainable AI</a>', 
  'YANG202229': '<a href="https://www.sciencedirect.com/science/article/pii/S1566253521001597" target="_blank">Unbox the black-box for the medical explainable AI via multi-modal and multi-centre data fusion&colon; A mini-review, two showcases and beyond</a> {cite}`YANG202229`', 
  'YamlMissingBattery': '<a href="https://realpython.com/python-yaml/" target="_blank">YAML</a>', 
  'ZHU202253': '<a href="https://www.sciencedirect.com/science/article/pii/S1566253521001548" target="_blank">Interpretable learning based Dynamic Graph Convolutional Networks for Alzheimer’s Disease analysis</a> {cite}`ZHU202253`', 
  'ZipImporter': '<a href="https://docs.python.org/3/library/zipimport.html#zipimporter-objects" target="_blank">zipimport.zipimporter</a>'
}
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
