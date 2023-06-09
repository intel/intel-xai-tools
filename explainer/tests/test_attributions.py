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
from deepdiff import DeepDiff
import numpy as np
import pandas as pd
import scipy as sp
import transformers
import torch
torch.manual_seed(0)
### library to be tested ###
from explainer import attributions
from attributions.plots import shap_waterwall_plot
from attributions.widgets import ShapUI
###################################

device = torch.device('cpu')

def test_deep_explainer(custom_pyt_CNN):
    model, test_loader, class_names = custom_pyt_CNN 
    X_test = next(iter(test_loader))[0].to(device)
    deViz = attributions.deep_explainer(model, X_test[:2], X_test[2:4], class_names)
    assert isinstance(deViz, attributions.attributions.DeepExplainer) 
    deViz.visualize()

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
