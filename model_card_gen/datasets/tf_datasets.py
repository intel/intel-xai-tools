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

from model_card_gen.datasets import BaseDataset

class TensorflowDataset(BaseDataset):
    """
    Class wrapper for Tensorflow tfrecord
    """
    def __init__(self, dataset_path, name=""):
        super().__init__(dataset_path, name)
        self._framework = "tensorflow"
    
    @property
    def framework(self):
        """
        Returns the framework for dataset
        """
        return self._framework
