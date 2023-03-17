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
import pandas as pd
import torch
### library to be tested ###
from explainer import metrics
###################################

# Namedtuple that holds necessary input and outputs to the cm and curve plots.
Data = namedtuple('Data', ['y_true', 'y_pred', 'label_names', 'recall', 'cm'])

@pytest.fixture(scope='session')
def simple_data():
    '''
    A simple 3-class use case with only 5 examples.

    '''
    y_true = [[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]]
    y_pred = [
        [.002, .09, .89],
        [.01, .7, .29],
        [.3, .67, .03],
        [.55, .4, .05],
        [.03, .86, .11]]
    label_names = ['cat', 'dog', 'horse']
    ### correct solutions ###
    recall = {
        0: np.array([1., 1., 0.5, 0.5, 0.5, 0.]),
        1: np.array([1., 1., 1., 0.5, 0.5, 0.]),
        2: np.array([1., 1., 1., 1., 1., 0.])
        }
    cm = pd.DataFrame(
        {label_names[0]: [0.5, 0.0, 0.0],
         label_names[1]: [0.5, 1.0, 0.0],
         label_names[2]: [0.0, 0.0, 1.0]},
         index=label_names)

    return Data(y_true, y_pred, label_names, recall, cm)

@pytest.mark.simple
def test_plot(simple_data):
    data = simple_data
    plotter = metrics.plot(data.y_true, data.y_pred, data.label_names)
    assert DeepDiff(plotter.recall, data.recall) == {}

@pytest.mark.simple
def test_confusion_matrix(simple_data):
    data = simple_data
    cm = metrics.confusion_matrix(data.y_true, data.y_pred, data.label_names)
    assert cm.df.equals(data.cm)

def test_confusion_matrix_pyt(custom_pyt_CNN):
    device = torch.device('cpu')
    model, test_loader, class_names = custom_pyt_CNN 

    # test the model
    model.eval()
    test_loss = 0
    correct = 0
    y_true = torch.empty(0)
    y_pred = torch.empty((0, 10))
    X_test = torch.empty((0, 1, 28, 28))

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            X_test = torch.cat((X_test, data))
            y_true, y_pred = torch.cat((y_true, target)), torch.cat((y_pred, output))
    
    # one-hot-encode for metrics explanation
    oh_y_true = torch.from_numpy(np.eye(int(torch.max(y_true)+1))[list(y_true.numpy().astype(int))])
    cm = metrics.confusion_matrix(oh_y_true, y_pred, class_names)
    assert isinstance(cm, metrics.metrics.ConfusionMatrix)
    cm.visualize()
