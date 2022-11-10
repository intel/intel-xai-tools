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
from typing import Text, Union, Optional
from model_card_gen.utils.types import DatasetType
from model_card_gen.analyze.analyzer import ModelAnalyzer
try:
    import torch
except: ImportError

class PTAnalyzer(ModelAnalyzer):
    def __init__(self,
                 model_path: Text,
                 dataset: DatasetType,
                 eval_config: Union[tfma.EvalConfig, Text] = None):
        """Start TFMA analysis on TensorFlow model
        Args:
            model_path (str) : path to model
            data (str): string ot tfrecord
            eval_config (tfma.EvalConfig pr str): representing proto file path
        """
        super().__init__(eval_config, dataset)
        self.dataset = dataset
        self.model_path = model_path
        self._model = self.load_model()
        self.eval_data = self.make_eval_dataframe()
    
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
            tfma.EvalResults()

        Example:
            >>> from model_card_gen.analyzer import PTAnalyzer
            >>> PTAnalyzer.analyze(
                model_path='compas/model',
                data='compas/eval.tfrecord',
                eval_config='compas/eval_config.proto')
        """
        self = cls(model_path, dataset, eval_config)
        self.run_analysis()
        return self.get_analysis()

    def load_model(self):
        model = torch.jit.load(self.model_path)
        model.eval()
        return model

    def predict_instances(self, instances):
        """
        Perform feed-forward inference and predict the classes of the input_samples
        """
        predictions = self._model(instances)
        # when model outputs binary classification
        if len(predictions.shape) >= 2 & predictions.shape[1] == 2:
            return predictions[:, 1]
        return predictions

    def get_inference_data(self):
        """
        Perform feed-forward inference and predict the classes of the input_samples
        """
        instances, true_values = self.dataset.dataset[:]
        predicted_values = self.predict_instances(instances)
        return instances, true_values, predicted_values.detach().numpy()

    def make_eval_dataframe(self):
        """
        Make pandas DataFrame for TFMA evaluation 
        """
        instances, true_values, predicted_values = self.get_inference_data()
        df = pd.DataFrame(instances, columns=self.dataset.feature_names)
        df['label'] = true_values
        df['prediction'] = predicted_values
        return df
    
    def run_analysis(self):
        self.eval_result = tfma.analyze_raw_data(data=self.eval_data,
                                                 eval_config=self.eval_config)
        return self.eval_result
