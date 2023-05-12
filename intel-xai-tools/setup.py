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

"""
XAI Tools, Explainer
"""
from pathlib import Path
from setuptools import setup

# read the contents of your README file
this_directory = Path(__file__).parent
long_description = (this_directory / "../README.md").read_text()

ATTRIBUTIONS_PKGS = [
  'captum',
  'intel-tensorflow==2.11.0',
  'numpy>=1.14.3,<1.23.0',
  'opencv-python',
  'scikit-plot',
  'scipy',
  'shap',
  'torch==1.13.1',
  'transformers',
]

CAM_PKGS = [
  'grad-cam',
  'matplotlib',
  'numpy>=1.14.3,<1.23.0',
  'opencv-python',
  'scipy',
  'torch==1.13.1',
]

METRICS_PKGS =  [
  'matplotlib',
  'pandas',
  'plotly',
  'scikit-learn',
  'seaborn',
]

MCG_PKGS = [
    'absl-py',
    'attrs>=19.3.0,<22',
    'dataclasses;python_version<"3.7"',
    'grpcio-status<1.49',
    'intel-tensorflow==2.11.0',
    'jinja2',
    'joblib>=1.2.0',
    'jsonschema[format-nongpl]',
    'plotly',
    'protobuf<3.20,>=3.9.2',
    'semantic-version',
    'tensorflow-data-validation',
    'tensorflow-model-analysis',
]

PYTORCH_PKGS = [
    'torch==1.13.1',
    'torchvision==0.14.1',
]

REQUIRED_PKGS =  (
  ATTRIBUTIONS_PKGS +
  CAM_PKGS +
  METRICS_PKGS +
  MCG_PKGS
)

TEST_PKGS = [
    'deepdiff',
    'pytest',
    'tensorflow-hub',
    'datasets',
]

PACKAGES = [
  'explainer',
  'explainer.attributions',
  'explainer.cam',
  'explainer.metrics',
  'explainer.utils.model',
  'model_card_gen',
  'model_card_gen.analyze',
  'model_card_gen.datasets',
  'model_card_gen.docs',
  'model_card_gen.docs.examples',
  'model_card_gen.graphics',
  'model_card_gen.utils',
]


EXTRAS = {
    'pytorch': PYTORCH_PKGS,
    'test': TEST_PKGS + PYTORCH_PKGS,
}

# Get version from version module.
with open('explainer/version.py') as fp:
  globals_dict = {}
  exec(fp.read(), globals_dict)  # pylint: disable=exec-used
__version__ = globals_dict['__version__']

setup(
    name='intel-xai',
    version=__version__,
    url='https://github.com/IntelAI/intel-xai-tools',
    license='Apache 2.0',
    author='IntelAI',
    author_email='IntelAI@intel.com',
    description='Intel® Explainable AI Tools',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=REQUIRED_PKGS,
    tests_require=TEST_PKGS,
    extras_require=EXTRAS,
    packages=PACKAGES,
    package_data={
        'model_card_gen': ['schema/**/*.json', 'template/**/*.jinja'],
        'model_card_gen.docs.examples': ['docs/examples/**/*.html',
                                         'docs/examples/**/*.json'],
    },
    include_package_data=True,
    zip_safe=False,
    platforms='any',
    classifiers=[
        # As from http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.9,<3.10',
    keywords='XAI, explainer',
)
