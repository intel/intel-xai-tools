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

from ..utils.model.model_framework import is_tf_model, is_pt_model, raise_unknown_model_error

class GradCAM:
    """GradCAM generator class. Depending on the model framework, GradCAM is a superclass to TFGradCAM or XGradCAM.
    Note that EigenCAM (only supports PyTorch) is not included yet.
    """
    def __new__(cls, model, *args):    
        from .tf_cam import TFGradCAM
        from .pt_cam import XGradCAM

        if is_tf_model(model):
            return super().__new__(TFGradCAM)
        elif is_pt_model(model):
            return super().__new__(XGradCAM)
        else:
            raise_unknown_model_error()
