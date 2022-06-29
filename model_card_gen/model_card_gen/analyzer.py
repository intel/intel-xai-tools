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

from typing import Text, Optional, Union, Sequence

import pandas as pd
import tensorflow_model_analysis as tfma
from google.protobuf import text_format

class ModelAnalyzer:
    def __init__(self,
            eval_config : Union[tfma.EvalConfig, Text] = None):
        """Start TFMA analysis
        Args:
            eval_config : tfma.EvalConfig or str representing proto file path
        Return:
            EvalResult
        """
        if isinstance(eval_config, tfma.EvalConfig):
            self.eval_config = eval_config
        elif isinstance(eval_config, str):
             self.eval_config = self.parse_eval_config(eval_config)
        else:
            raise ValueError("ModelAnalyzer needs either eval_config or eval_config_path argument.")
        self.eval_result = None

    @classmethod
    def analyze(cls,
            model_path: Optional[Text] = '',
            eval_config : Union[tfma.EvalConfig, Text] = None,
            data : Sequence[Union[Text, pd.DataFrame]] = ''):
        """Class Factory to start TFMA analysis
        Args:
            eval_config : tfma.EvalConfig or str representing proto file path
        
        Returns:
            tfma.EvalConfig()
        """
        if bool(data) and isinstance(data, dict) and 'eval' in data:
            self = TFAnalyzer(model_path, data['eval'], eval_config)
        elif bool(data) and all(isinstance(elem, str) for elem in data):
            self = TFAnalyzer(model_path, data[0], eval_config)
        elif isinstance(data, str):
            self = TFAnalyzer(model_path, data, eval_config)
        elif isinstance(data, pd.DataFrame):
            self = DFAnalyzer(data, eval_config)
        elif bool(data) and all(isinstance(elem, pd.DataFrame) for elem in data):
            self = DFAnalyzer(data[0], eval_config)
        elif not bool(data):
            raise ValueError("ModelAnalyzer needs data argument.")
        self.analyze()
        return self.get()

    def parse_eval_config(self, eval_config_path):
        """Parse proto file from file path to generate Eval config

        Args:
            eval_config_path : str representing proto file path
        
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
                 raw_data : pd.DataFrame,
                 eval_config : Union[tfma.EvalConfig, Text] = None):
        """Start TFMA analysis on Pandas DataFrame
        
        Args:
            raw_data : pd.DataFrame
            eval_config : tfma.EvalConfig or str representing proto file path
        """
        super().__init__(eval_config)
        self.raw_data = raw_data

    def analyze(self):
        self.eval_result = tfma.analyze_raw_data(data=self.raw_data,
                              eval_config=self.eval_config)
        return self.eval_result

class TFAnalyzer(ModelAnalyzer):
    def __init__(self,
                 model_path : Text,
                 data_path : Text,
                 eval_config : Union[tfma.EvalConfig, Text] = None):
        super().__init__(eval_config)
        self.model_path = model_path
        self.data_path = data_path

    def analyze(self):
        #TODO if not eval_shared
        eval_shared_model = tfma.default_eval_shared_model(
            eval_saved_model_path=self.model_path,
            eval_config=self.eval_config)
        
        self.eval_result = tfma.run_model_analysis(
            eval_shared_model=eval_shared_model,
            eval_config=self.eval_config,
            data_location=self.data_path)
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
