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
  'intel-tensorflow==2.9.1',
  'intel-scipy==1.7.3',
  'captum==0.5.0',
  'shap @ git+https://github.com/slundberg/shap@v0.41.0',
  'scikit-plot==0.3.7',
  'transformers==4.20.1',
  'torch==1.13.0',
  'opencv-python==4.6.0.66',
]

CAM_PKGS = [
  'grad-cam==1.4.6',
  'matplotlib==3.6.2',
  'numpy==1.23.5',
  'opencv-python==4.6.0.66',
  'torch==1.13.0',
  'scipy==1.10.0',
]

METRICS_PKGS =  [
  'matplotlib~=3.6.0',
  'seaborn==0.12.0',
  'scikit-learn~=1.1.2',
  'pandas==1.5.0',
  'plotly==5.10.0',
  'jupyter-plotly-dash==0.4.3',
]

REQUIRED_PACKAGES =  ATTENTION_PKGS + ATTRIBUTIONS_PKGS + CAM_PKGS + METRICS_PKGS

PACKAGES = [
  "explainer",
  "explainer.attention_layers",
  "explainer.cam",
  "explainer.attributions",
  "explainer.metrics",
]

TEST_PACKAGES = [
    'pytest'
]

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
    install_requires=REQUIRED_PACKAGES,
    tests_require=TEST_PACKAGES,
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
