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
from typing import Text, Optional, Union
from model_card_gen.utils.types import DatasetType
from model_card_gen.analyze.analyzer import ModelAnalyzer

class TFAnalyzer(ModelAnalyzer):
    def __init__(self,
                 model_path: Text,
                 dataset: DatasetType,
                 eval_config: Union[tfma.EvalConfig, Text] = None):
        """Start TFMA analysis on TensorFlow model
        Args:
            model_path (str) : path to model
            data (str): tfrecord file glob path
            eval_config (tfma.EvalConfig pr str): representing proto file path
        """
        super().__init__(eval_config, dataset)
        self.model_path = model_path
        self.dataset = dataset
    
    @classmethod
    def analyze(cls,
                model_path: Optional[Text] = '',
                eval_config: Union[tfma.EvalConfig, Text] = None,
                dataset:  DatasetType = None):
        """Class Factory to start TFMA analysis
        Args:
            model_path (str) : path to model
            eval_config (tfma.EvalConfig or str): representing proto file path
            data (str or pd.DataFrame): string ot tfrecord or raw dataframe containing
                prediction values and  ground truth

        Raises:
            TypeError: when eval_config is not of type tfma.EvalConfig or str
            TypeError: when data argument is not of type pd.DataFrame or str

        Returns:
            tfma.EvalConfig()

        Example:
            >>> from model_card_gen.analyzer import TFAnalyzer
            >>> TFAnalyzer.analyze(
                model_path='compas/model',
                data='compas/eval.tfrecord',
                eval_config='compas/eval_config.proto')
        """
        self = cls(model_path, dataset, eval_config)
        self.run_analysis()
        return self.get_analysis()
    
    def run_analysis(self):
        # TODO if not eval_shared
        eval_shared_model = tfma.default_eval_shared_model(
            eval_saved_model_path=self.model_path,
            eval_config=self.eval_config)

        self.eval_result = tfma.run_model_analysis(
            eval_shared_model=eval_shared_model,
            eval_config=self.eval_config,
            data_location=self.dataset.dataset_path)
        return self.eval_result
