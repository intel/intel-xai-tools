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

from distutils.command import build
from setuptools import setup

REQUIRED_PACKAGES = [
    'absl-py>=1.0.0',
    'apache-beam==2.41.0',
    'attrs<22,>=19.3.0',
    'dataclasses;python_version<"3.7"',
    'intel-tensorflow<2.12.0',
    'jinja2>=3,<4',
    'joblib>=1.2.0',
    'jsonschema[format-nongpl]>=4.3.0',
    'plotly>=3.8.1,<6',
    'protobuf<3.20,>=3.9.2',
    'pyarrow<7,>=6',
    'semantic-version>=2.8.0,<3',
    'tensorflow-data-validation>=1.11.0,<1.12.0',
    'tensorflow-model-analysis>=0.42.0,<0.43.0'
]

TEST_PACKAGES = [
    'httplib2<0.19.1',
    'pytest',
    'tensorflow-hub'
]

PYTORCH_PACKAGES = [
    'torch<1.14'
]

EXTRAS = {
    'test': TEST_PACKAGES + PYTORCH_PACKAGES,
    'pytorch': PYTORCH_PACKAGES,
}

# Get version from version module.
with open('model_card_gen/version.py') as fp:
  globals_dict = {}
  exec(fp.read(), globals_dict)  # pylint: disable=exec-used
__version__ = globals_dict['__version__']

with open('README.md', 'r', encoding='utf-8') as fh:
  _LONG_DESCRIPTION = fh.read()


setup(
    name='model-card-generator',
    version=__version__,
    description='Model Card Generator',
    long_description=_LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url='',
    author='IntelAI',
    author_email='IntelAI@intel.com',
    packages=[
        'model_card_gen',
        'model_card_gen.analyze',
        'model_card_gen.datasets',
        'model_card_gen.docs',
        'model_card_gen.docs.examples',
        'model_card_gen.graphics',
        'model_card_gen.utils',
    ],
    package_data={
        'model_card_gen': ['schema/**/*.json', 'template/**/*.jinja'],
        'model_card_gen.docs.examples': ['docs/examples/**/*.html',
                                         'docs/examples/**/*.json'],
    },
    python_requires='>=3.6,<4',
    install_requires=REQUIRED_PACKAGES,
    tests_require=TEST_PACKAGES,
    extras_require=EXTRAS,
    # PyPI package labels.
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='',
    keywords='model card',
)
