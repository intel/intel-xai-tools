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


class BaseDataset:
    """
    Base class for all datasets supported by ModelCardGen
    """
    def __init__(self, dataset_path=None, name=None):
        self._dataset_path = dataset_path
        self._dataset_name = name

    @property
    def dataset_path(self):
        """
        Returns the file path of the dataset
        """
        return self._dataset_path

    @property
    def name(self):
        """
        Returns the name of the dataset
        """
        return self._name
    
    @property
    def description(self):
        """
        Returns the description of the dataset
        """
        return self._description
