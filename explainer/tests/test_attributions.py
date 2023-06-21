#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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

### libraries to support tests ###
import pytest
import numpy as np
import pandas as pd
import torch
import scipy as sp
import transformers
torch.manual_seed(0)
### library to be tested ###
from explainer import attributions
from attributions.plots import shap_waterwall_plot
from attributions.widgets import ShapUI
###################################

@pytest.mark.parametrize("custom_CNN", ['custom_pyt_CNN', 'custom_tf_CNN'])
def test_DeepExplainer(custom_CNN, request):
    '''
    Test every way to instantiate DeepExplainer and PTDeepExplainer objects
    '''
    model, X_test, class_names, y_test = request.getfixturevalue(custom_CNN)
    if isinstance(X_test, torch.Tensor) == True:
        deViz = attributions.DeepExplainer(model, X_test[:2], X_test[2:4], class_names)
        assert isinstance(deViz, attributions.PTDeepExplainer) 
        deViz = attributions.PTDeepExplainer(model, X_test[:2], X_test[2:4], class_names)
        assert isinstance(deViz, attributions.PTDeepExplainer) 
        deViz = attributions.deep_explainer(model, X_test[:2], X_test[2:4], class_names)
        assert isinstance(deViz, attributions.PTDeepExplainer) 
    else:
        deViz = attributions.DeepExplainer(model, X_test[:2], X_test[2:4], class_names)
        assert isinstance(deViz, attributions.DeepExplainer) 
        deViz = attributions.deep_explainer(model, X_test[:2], X_test[2:4], class_names)
        assert isinstance(deViz, attributions.DeepExplainer) 
    deViz.visualize()

@pytest.mark.parametrize("custom_CNN", ['custom_pyt_CNN', 'custom_tf_CNN'])
def test_DeepExplainer_one_image(custom_CNN, request):
    '''
    Test edge case of one image input (with 1 and 2 background images)
    '''
    model, X_test, class_names, y_test = request.getfixturevalue(custom_CNN)
    deViz = attributions.deep_explainer(model, X_test[:1], X_test[2:3], class_names)
    deViz = attributions.deep_explainer(model, X_test[:1], X_test[2:4], class_names)
    deViz.visualize()

@pytest.mark.parametrize("custom_CNN", ['custom_pyt_CNN', 'custom_tf_CNN'])
def test_GradientExplainer(custom_CNN, request):
    '''
    Test every way to instantiate GradientExplainer and PTGradientExplainer objects
    '''
    model, X_test, class_names, y_test = request.getfixturevalue(custom_CNN)
    if isinstance(X_test, torch.Tensor) == True:
        geViz = attributions.GradientExplainer(model, X_test[:2], X_test[2:4], class_names)
        assert isinstance(geViz, attributions.PTGradientExplainer) 
        geViz = attributions.PTGradientExplainer(model, X_test[:2], X_test[2:4], class_names)
        assert isinstance(geViz, attributions.PTGradientExplainer) 
        geViz = attributions.gradient_explainer(model, X_test[:2], X_test[2:4], class_names)
        assert isinstance(geViz, attributions.PTGradientExplainer) 
    else:
        geViz = attributions.GradientExplainer(model, X_test[:2], X_test[2:4], class_names)
        assert isinstance(geViz, attributions.GradientExplainer) 
        geViz = attributions.gradient_explainer(model, X_test[:2], X_test[2:4], class_names)
        assert isinstance(geViz, attributions.GradientExplainer) 
    geViz.visualize()

@pytest.mark.parametrize("custom_CNN", ['custom_pyt_CNN', 'custom_tf_CNN'])
def test_GradientExplainer_one_image(custom_CNN, request):
    '''
    Test edge case of one image input (with 1 and 2 background images and 1 and 2 ranked outputs)
    '''
    model, X_test, class_names, y_test = request.getfixturevalue(custom_CNN)
    geViz = attributions.gradient_explainer(model, X_test[:1], X_test[2:3], class_names, 1)
    geViz = attributions.gradient_explainer(model, X_test[:1], X_test[2:3], class_names, 2)
    geViz = attributions.gradient_explainer(model, X_test[:1], X_test[2:4], class_names, 1)
    geViz = attributions.gradient_explainer(model, X_test[:1], X_test[2:4], class_names, 2)
    geViz.visualize()

def test_partition_image(tf_resnet50, dog_cat_image, imagenet_class_names):
    from tensorflow.keras.applications.resnet50 import preprocess_input
    def f(X):
        tmp = X.copy()
        preprocess_input(tmp)
        return tf_resnet50(tmp)

    image = np.expand_dims(dog_cat_image, axis=0) 

    # test generator class manually
    pe = attributions.PartitionExplainer('image', f, imagenet_class_names, image[0].shape)
    pe.run_explainer(image)
    assert isinstance(pe, attributions.PartitionImageExplainer)
    pe.visualize()

    # test explicit class
    pe = attributions.PartitionImageExplainer('image', f, imagenet_class_names,image[0].shape)
    pe.run_explainer(image)
    assert isinstance(pe, attributions.PartitionImageExplainer)
    pe.visualize()

    # test module function
    pe = attributions.partition_image_explainer(f, imagenet_class_names, image)
    assert isinstance(pe, attributions.PartitionImageExplainer)
    pe.visualize()

def test_partition_text():
    labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    tokenizer = transformers.AutoTokenizer.from_pretrained("nateraw/bert-base-uncased-emotion", use_fast=True)
    model = transformers.AutoModelForSequenceClassification.from_pretrained("nateraw/bert-base-uncased-emotion")

    def f(x):
        tv = torch.tensor([tokenizer.encode(v, padding='max_length', max_length=128, truncation=True) for v in x])
        attention_mask = (tv!=0).type(torch.int64)
        outputs = model(tv,attention_mask=attention_mask)[0].detach().cpu().numpy()
        scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
        val = sp.special.logit(scores)
        return val
    
    # test generator class manually
    pe = attributions.PartitionExplainer('text', f, labels, tokenizer)
    pe.run_explainer(['i didnt feel humiliated'])
    assert isinstance(pe, attributions.PartitionTextExplainer)
    pe.visualize()

    # test explicit class 
    pe = attributions.PartitionTextExplainer('text', f, labels, tokenizer)
    pe.run_explainer(['i didnt feel humiliated'])
    assert isinstance(pe, attributions.PartitionTextExplainer)
    pe.visualize()

    # test module function
    pe = attributions.partition_text_explainer(f, labels, ['i didnt feel humiliated'], tokenizer)
    assert isinstance(pe, attributions.PartitionTextExplainer)
    pe.visualize()


def test_shap_waterwall_plot():
    kwargs = dict(expected_value=.5,
                  shap_values=np.random.normal(0, 1, 10),
                  feature_values=np.random.rand(10),
                  columns=list(range(10)),
                  y_true=0)

    assert shap_waterwall_plot(**kwargs)


def test_shap_ui():
    kwargs = dict(df=pd.DataFrame({"Feature 1": np.random.rand(10), 
                                   "Feature 2": np.random.rand(10)}),
                  shap_values=np.random.normal(0, 1, (10, 2)),
                  expected_value=.5,
                  y_true=np.random.randint(0, 2, 10),
                  y_pred=np.random.randint(0, 2, 10))

    ui = ShapUI(**kwargs)
    ui.show()
    assert ui.view
