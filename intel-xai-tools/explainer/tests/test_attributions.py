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
from collections import namedtuple
import pytest
from deepdiff import DeepDiff
import numpy as np
import torch, torchvision
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
import scipy as sp
import transformers
torch.manual_seed(0)
### library to be tested ###
from explainer import attributions
###################################

device = torch.device('cpu')

def test_deep_explainer(custom_pyt_CNN):
    model, test_loader, class_names = custom_pyt_CNN 
    X_test = next(iter(test_loader))[0].to(device)
    deViz = attributions.deep_explainer(model, X_test[:2], X_test[2:4], class_names)
    assert isinstance(deViz, attributions.attributions.DeepExplainer) 
    deViz.visualize()

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