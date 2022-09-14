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
# SPDX-License-Identifier: EPL-2.0
#

import pandas as pd
import tensorflow_model_analysis as tfma
from google.protobuf import text_format
from typing import Text, Optional, Union, get_args
from model_card_gen.utils.types import DatasetType
from model_card_gen.datasets import TensorflowDataset, PytorchDataset
import torch 

class ModelAnalyzer:
    def __init__(self,
            eval_config : Union[tfma.EvalConfig, Text] = None,
            dataset : DatasetType = None):
        """Start TFMA analysis
        Args:
            eval_config (tfma.EvalConfig or str): representing proto file path
        Return:
            EvalResult
        """
        self.eval_config: tfma.EvalConfig = self.check_eval_config(eval_config)
        self.dataset: DatasetType = self.check_data(dataset)

    @classmethod
    def analyze(cls,
            model_path: Optional[Text] = '',
            eval_config : Union[tfma.EvalConfig, Text] = None,
            dataset :  DatasetType = None):
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
            >>> from model_card_gen.analyzer import ModelAnalyzer
            >>> ModelAnalyzer.analyze(
                model_path='compas/model',
                data='compas/eval.tfrecord',
                eval_config='compas/eval_config.proto')
        """
        self = cls(eval_config, dataset)
        if isinstance(dataset, TensorflowDataset):
            self = TFAnalyzer(model_path, dataset, eval_config)
        elif isinstance(dataset, pd.DataFrame):
            self = DFAnalyzer(dataset, eval_config)
        elif isinstance(dataset, PytorchDataset):
            self = PTAnalyzer(model_path, dataset, eval_config)
        self.analyze()
        return self.get()

    def check_eval_config(self, eval_config) -> tfma.EvalConfig:
        """Check that eval_config argument is of type tfma.EvalConfig or str"""
        if isinstance(eval_config, tfma.EvalConfig):
            return eval_config
        elif isinstance(eval_config, str):
             return self.parse_eval_config(eval_config)
        else:
            raise TypeError("ModelAnalyzer requres eval_config argument of type tfma.EvalConfig or str.")

    def check_data(self, dataset):
        """Check that data argument is of type pd.DataFrame or DatasetType"""
        if not (isinstance(dataset, get_args(DatasetType)) or isinstance(dataset, pd.DataFrame)):
            raise TypeError("ModelAnalyzer.analyze requires data argument to be of type pd.DataFrame, TensorflowDataset or PytorchDataset")
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

    def get(self):
        """Retrieve eval_results attribute
        """
        return self.eval_result

class DFAnalyzer(ModelAnalyzer):
    def __init__(self,
                 dataset : pd.DataFrame,
                 eval_config : Union[tfma.EvalConfig, Text] = None):
        """Start TFMA analysis on Pandas DataFrame
        
        Args:
            raw_data (pd.DataFrame): dataframe containing prediciton values and ground truth
            eval_config (tfma.EvalConfig or str): representing proto file path
        """
        super().__init__(eval_config, dataset)

    def analyze(self):
        self.eval_result = tfma.analyze_raw_data(data=self.dataset.dataset_path,
                              eval_config=self.eval_config)
        return self.eval_result

class TFAnalyzer(ModelAnalyzer):
    def __init__(self,
                 model_path : Text,
                 dataset : DatasetType,
                 eval_config : Union[tfma.EvalConfig, Text] = None):
        """Start TFMA analysis on TensorFlow model
        Args:
            model_path (str) : path to model
            data (str): string ot tfrecord
            eval_config (tfma.EvalConfig pr str): representing proto file path
        """
        super().__init__(eval_config, dataset)
        self.model_path = model_path
        self.dataset = dataset

    def analyze(self):
        #TODO if not eval_shared
        eval_shared_model = tfma.default_eval_shared_model(
            eval_saved_model_path=self.model_path,
            eval_config=self.eval_config)

        self.eval_result = tfma.run_model_analysis(
            eval_shared_model=eval_shared_model,
            eval_config=self.eval_config,
            data_location=self.dataset.dataset_path)
        return self.eval_result

def analyze_model(model_path: Optional[Text] = '',
            eval_config : Union[tfma.EvalConfig, Text] = None,
            data : Union[Text, pd.DataFrame] = '') -> Union[DFAnalyzer, TFAnalyzer] :
        if isinstance(data, str):
            cls = TFAnalyzer(model_path, data, eval_config)
        elif isinstance(data, pd.DataFrame):
            cls = DFAnalyzer(data, eval_config)
        cls.analyze()
        return cls

class PTAnalyzer(ModelAnalyzer):
    def __init__(self,
                 model_path : Text,
                 dataset : DatasetType,
                 eval_config : Union[tfma.EvalConfig, Text] = None):
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

    def load_model(self):
        model = torch.jit.load(self.model_path)
        model.eval()
        return model
    
    def predict_instances(self, instances):
        """
        Perform feed-forward inference and predict the classes of the input_samples
        """
        predictions = self._model(instances)
        _, predicted_ids = torch.max(predictions, dim=1)
        return predicted_ids
    
    def get_inference_data(self):
        """
        Perform feed-forward inference and predict the classes of the input_samples
        """
        instances, true_values = self.dataset.dataset[:]
        predicted_values = self.predict_instances(instances)
        return instances, true_values, predicted_values
    
    def make_eval_dataframe(self):
        """
        Make pandas DataFrame for TFMA evaluation 
        """
        instances, true_values, predicted_values = self.get_inference_data()        
        df = pd.DataFrame(instances, columns=self.dataset.feature_names)
        df['true_values'] = true_values
        df['predicted_values'] = predicted_values
        return df

    def analyze(self):
        self.eval_result = tfma.analyze_raw_data(data=self.eval_data,
                              eval_config=self.eval_config)
        return self.eval_result