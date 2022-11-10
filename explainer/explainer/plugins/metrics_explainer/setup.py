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

from setuptools import setup

setup(
    name='explainer-explainers-metrics_explainer',
    version='0.1',
    zip_safe=False,
    platforms='any',
    py_modules=['metrics_explainer'],
    include_package_data=True,
    install_requires=[
        'matplotlib==3.6.0',
        'seaborn==0.12.0',
        'scikit-learn==1.1.2',
        'pandas==1.5.0',
        'plotly==5.10.0',
        'jupyter-plotly-dash==0.4.3',
    ],
    entry_points={ 
        'explainer.explainers.metrics_explainer': [
            'confusionmatrix= metrics_explainer:confusion_matrix [groundtruth,predictions,labels]',
            'plot=metrics_explainer:plot [groundtruth,predictions,labels]',
            'pstats=metrics_explainer:pstats [command]',
        ]
    }, 
    python_requires='>=3.9,<3.10'
)
