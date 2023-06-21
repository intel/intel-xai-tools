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
from typing import Text, Union
from model_card_gen.utils.types import DatasetType
from model_card_gen.analyze.analyzer import ModelAnalyzer

class DFAnalyzer(ModelAnalyzer):
    def __init__(self,
                 eval_config: Union[tfma.EvalConfig, Text] = None,
                 dataset: pd.DataFrame = None):
        """Start TFMA analysis on Pandas DataFrame

        Args:
            raw_data (pd.DataFrame): dataframe containing prediciton values and ground truth
            eval_config (tfma.EvalConfig or str): representing proto file path
        """
        super().__init__(eval_config, dataset)
    
    @classmethod
    def analyze(cls,
                eval_config: Union[tfma.EvalConfig, Text] = None,
                dataset:  DatasetType = None,):
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
            tfma.EvalResults()

        Example:
            >>> from model_card_gen.analyzer import DFAnalyzer
            >>> DFAnalyzer.analyze(
                model_path='compas/model',
                data='compas/eval.tfrecord',
                eval_config='compas/eval_config.proto')
        """
        self = cls(eval_config, dataset)
        self.run_analysis()
        return self.get_analysis()

    def run_analysis(self):
        self.eval_result = tfma.analyze_raw_data(data=self.dataset,
                                                 eval_config=self.eval_config)
        return self.eval_result
