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
    name='explainer-explainers-lm-layers-explainer',
    version='0.1',
    zip_safe=False,
    platforms='any',
    py_modules=['lm_layers_explainer'],
    include_package_data=True,
    install_requires=[
        'matplotlib==3.3.1',
        'bertviz==1.4.0',
        'ecco==0.1.2',
    ],
    entry_points={ 
        'explainer.explainers.lm_layers_explainer': [
            'activations = lm_layers_explainer:activations [model,text]',
            'attention_head_view = lm_layers_explainer:attention_head_view [model]',
            'attention_model_view = lm_layers_explainer:attention_model_view [model]',
            'attention_neuron_view = lm_layers_explainer:attention_neuron_view [model]',
            'attributions = lm_layers_explainer:attributions [model]',
            'predictions = lm_layers_explainer:predictions [model,text]',
            'rankings = lm_layers_explainer:rankings [model]',
        ]
    }, 
    python_requires='>=3.9,<3.10'
)
