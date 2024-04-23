#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Intel Corporation
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

from intel_ai_safety.common.frameworks.model_framework import get_model_framework
from intel_ai_safety.common.plugins import get_plugin_extended_cls
from intel_ai_safety.explainer.base_explainer import BaseExplainer

from intel_ai_safety.common.constants import ModelFramework

class GradCAM(BaseExplainer):
    """GradCAM generator class. Depending on the model framework, GradCAM is a
    superclass to TFGradCAM or XGradCAM. Note that EigenCAM (only supports
    PyTorch) is not included yet.
    """
    def __new__(cls, model, *args):
        modle_framework = get_model_framework(model)
        if modle_framework is ModelFramework.TENSORFLOW:
            TFGradCAM = get_plugin_extended_cls('explainer.cam.tf_cam.TFGradCAM')
            return super().__new__(TFGradCAM)
        elif modle_framework is ModelFramework.PYTORCH:
            XGradCAM = get_plugin_extended_cls('explainer.cam.pt_cam.XGradCAM')
            return super().__new__(XGradCAM)