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

import pandas as pd
import tensorflow_model_analysis as tfma
from google.protobuf import text_format
from typing import Text, Union, get_args
from model_card_gen.utils.types import DatasetType

class ModelAnalyzer:
    def __init__(self,
                 eval_config: Union[tfma.EvalConfig, Text] = None,
                 dataset: DatasetType = None):
        """Start TFMA analysis
        Args:
            eval_config (tfma.EvalConfig or str): representing proto file path
        Return:
            EvalResult
        """
        self.eval_config: tfma.EvalConfig = self.check_eval_config(eval_config)
        self.dataset: DatasetType = self.check_data(dataset)

    def check_eval_config(self, eval_config) -> tfma.EvalConfig:
        """Check that eval_config argument is of type tfma.EvalConfig or str"""
        if isinstance(eval_config, tfma.EvalConfig):
            return eval_config
        elif isinstance(eval_config, str):
            return self.parse_eval_config(eval_config)
        else:
            raise TypeError(
                "ModelAnalyzer requres eval_config argument of type tfma.EvalConfig or str.")

    def check_data(self, dataset):
        """Check that data argument is of type pd.DataFrame or DatasetType"""
        if not (isinstance(dataset, get_args(DatasetType)) or isinstance(dataset, pd.DataFrame)):
            raise TypeError(
                "ModelAnalyzer.analyze requires data argument to be of type pd.DataFrame, TensorflowDataset or PytorchDataset")
        return dataset

    def parse_eval_config(self, eval_config_path):
        """Parse proto file from file path to generate Eval config

        Args:
            eval_config_path (str): representing proto file path

        Returns:
            tfma.EvalConfig()
        """
        with open(eval_config_path, 'r') as f:
            eval_config_str = f.read()
        eval_config = text_format.Parse(eval_config_str, tfma.EvalConfig())
        return eval_config

    def get_analysis(self):
        """Retrieve eval_results attribute
        """
        return self.eval_result
