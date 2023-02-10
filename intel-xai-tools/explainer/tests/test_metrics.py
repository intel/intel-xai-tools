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

### libraries to support tests ###
import pytest
from deepdiff import DeepDiff
import numpy as np
import pandas as pd
### library to be tested ###
from explainer import metrics
###################################

class Data:
    '''
    Data class that holds necessary input and outputs to metrics_explainer
    plugins.
    '''
    def __init__(self, y_true, y_pred, label_names, recall, cm):
        # groundtruth values
        self.y_true = y_true
        # prediction values
        self.y_pred = y_pred
        # class names
        self.label_names = label_names

        ### solutions to tests ###
        self.recall = recall
        self.cm = cm

@pytest.fixture
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
def test_confusionmatrix(simple_data):
    data = simple_data
    cm = metrics.confusion_matrix(data.y_true, data.y_pred, data.label_names)
    assert cm.df.equals(data.cm)