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
    name='explainer-explainers-feature-attributions-explainer',
    version='0.2',
    zip_safe=False,
    platforms='any',
    py_modules=['feature_attributions_explainer'],
    include_package_data=True,
    install_requires=[
        'intel-tensorflow==2.9.1',
        'intel-scipy==1.7.3',
        'captum==0.5.0',
        'shap @ git+https://github.com/slundberg/shap@v0.41.0',
        'scikit-learn==1.1.2',
        'scikit-plot==0.3.7',
        'transformers==4.20.1',
        'torch==1.12.0',
        'opencv-python==4.6.0.66',
    ],
    entry_points={ 
        'explainer.explainers.feature_attributions_explainer': [
            'explainer = feature_attributions_explainer:explainer',
            'kernelexplainer = feature_attributions_explainer:kernel_explainer [model,data]',
            'deepexplainer = feature_attributions_explainer:deep_explainer [model,backgroundImages,targetImages,labels]',
            'gradientexplainer = feature_attributions_explainer:gradient_explainer [model,backgroundImages,targetImages,rankedOutputs,labels]',
            'partitionexplainer = feature_attributions_explainer:partition_explainer [model,tokenizer,categories]',
            'integratedgradients = feature_attributions_explainer:integratedgradients [model]',
            'deeplift = feature_attributions_explainer:deeplift [model]',
            'smoothgrad = feature_attributions_explainer:smoothgrad [model]',
            'featureablation = feature_attributions_explainer:featureablation [model]',
            'saliency = feature_attributions_explainer:saliency [model]',
            'sentimentanalysis = feature_attributions_explainer:sentiment_analysis [model, text]',
        ]
    }, 
    python_requires='>=3.9,<3.10'
)
