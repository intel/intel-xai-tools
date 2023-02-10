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
from setuptools import setup

ATTENTION_PKGS = ['bertviz~=1.4.0',
]

ATTRIBUTIONS_PKGS = [
  'intel-tensorflow<2.12.0',
  'scipy==1.10.0',
  'captum==0.5.0',
  'shap @ git+https://github.com/slundberg/shap@v0.41.0',
  'scikit-plot==0.3.7',
  'transformers==4.20.1',
  'torch<1.14.0',
  'opencv-python==4.6.0.66',
]

CAM_PKGS = [
  'grad-cam==1.4.6',
  'matplotlib==3.6.2',
  'numpy<1.23.0,>=1.17',
  'opencv-python==4.6.0.66',
  'torch<1.14.0',
  'scipy==1.10.0',
]

METRICS_PKGS =  [
  'matplotlib~=3.6.0',
  'seaborn==0.12.0',
  'scikit-learn~=1.1.2',
  'pandas~=1.5.0',
  'plotly>=3.8.1,<6',
]

MCG_PKGS = [
    'absl-py>=1.0.0',
    'attrs<22,>=19.3.0',
    'intel-tensorflow<2.12.0',
    'dataclasses;python_version<"3.7"',
    'jinja2>=3,<4',
    'joblib>=1.2.0',
    'jsonschema[format-nongpl]>=4.3.0',
    'plotly>=3.8.1,<6',
    'semantic-version>=2.8.0,<3',
    'tensorflow-data-validation>=1.11.0,<1.12.0',
    'tensorflow-model-analysis>=0.42.0,<0.43.0',
]

PYTORCH_PKGS = [
    'torch<1.14',
]

REQUIRED_PKGS =  (
  ATTENTION_PKGS + 
  ATTRIBUTIONS_PKGS + 
  CAM_PKGS +
  METRICS_PKGS +
  MCG_PKGS
)

TEST_PKGS = [
    'pytest',
    'tensorflow-hub',
    'deepdiff'
]

PACKAGES = [
  "explainer",
  "explainer.attention_layers",
  "explainer.cam",
  "explainer.attributions",
  "explainer.metrics",
  'model_card_gen',
  'model_card_gen.analyze',
  'model_card_gen.datasets',
  'model_card_gen.docs',
  'model_card_gen.docs.examples',
  'model_card_gen.graphics',
  'model_card_gen.utils',
]


EXTRAS = {
    'test': TEST_PKGS + PYTORCH_PKGS,
    'pytorch': PYTORCH_PKGS,
}

# Get version from version module.
with open('explainer/version.py') as fp:
  globals_dict = {}
  exec(fp.read(), globals_dict)  # pylint: disable=exec-used
__version__ = globals_dict['__version__']

setup(
    name='intel-xai-tools',
    version=__version__,
    url='https://github.com/IntelAI/intel-xai-tools',
    license='Apache 2.0',
    author='IntelAI',
    author_email='IntelAI@intel.com',
    description='Explainer invokes an explainer given a model, dataset and features',
    long_description=__doc__,
    install_requires=REQUIRED_PKGS,
    tests_require=TEST_PKGS,
    extras_require=EXTRAS,
    packages=PACKAGES,
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
        'License :: OSI Approved :: Apache 2.0 License',
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
