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
# SPDX-License-Identifier: EPL-2.0
#
"""
XAI Tools, Explainer
"""
from setuptools import setup, find_packages

dependencies = [
  'click~=8.1.3',
  'click-completion~=0.5.2',
  'pyyaml~=6.0',
  'urllib3[secure]~=1.26.11'
]

include_modules = [
  "explainer",
  "explainer.api",
  "explainer.cli",
  "explainer.version",
  "explainer.commands",
  "explainer.commands.*",
]

test_dependencies = [
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
    packages=find_packages(include=include_modules),
    include_package_data=True,
    zip_safe=False,
    platforms='any',
    install_requires=dependencies,
    tests_require=test_dependencies,
    entry_points={
        'console_scripts': [
            'explainer = explainer.cli:cli',
        ],
    },
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
