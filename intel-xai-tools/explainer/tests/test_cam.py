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
import torch 
torch.manual_seed(0)
### library to be tested ###
from explainer import cam 
###################################

device = torch.device('cpu')

def test_grad_cam(custom_pyt_CNN):
    model, test_loader, class_names = custom_pyt_CNN 
    X_test = next(iter(test_loader))[0].to(device)[0]
    image = torch.movedim(X_test, 0, 2).numpy()
    target_layer = model.conv_layers
    # use the highest-scoring category as the target class
    target_class = None
    image_dims = (28, 28)
    gcam = cam.xgradcam(model, target_layer, target_class, image, image_dims, device)
    assert isinstance(gcam, cam.XGradCAM)
    gcam.visualize()
