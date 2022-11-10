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
from typing import Text, Union, Optional, Dict
from model_card_gen.utils.types import DatasetType
from model_card_gen.analyze import TFAnalyzer, PTAnalyzer, DFAnalyzer 
from model_card_gen.datasets import TensorflowDataset, PytorchDataset

def get_analyzers(model_path: Optional[Text] = '',
                eval_config: Union[tfma.EvalConfig, Text] = None,
                datasets:  DatasetType = None):
    """Helper function to to get colleciton of analyzer objects
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
        >>> from model_card_gen.analyze import get_analyzers
        >>> get_analyzers(model_path='compas/model',
                         data='compas/eval.tfrecord',
                         eval_config='compas/eval_config.proto')
    """
    if all(isinstance(ds, TensorflowDataset) for ds in datasets.values()):
        analyzers = (TFAnalyzer(model_path, dataset, eval_config)
                     for dataset in datasets.values())
    elif all(isinstance(ds, pd.DataFrame) for ds in datasets.values()):
        analyzers = (DFAnalyzer(dataset, eval_config)
                     for dataset in datasets.values())
    elif all(isinstance(ds, PytorchDataset) for ds in datasets.values()):
        analyzers = (PTAnalyzer(model_path, dataset, eval_config)
                     for dataset in datasets.values())
    else:
        analyzers = ()
    return analyzers

def get_analysis(model_path: Optional[Text] = '',
                eval_config: Union[tfma.EvalConfig, Text] = None,
                datasets:  Dict = {}):
    """Helper function to run TFMA analysis for collection of datasets
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
        >>> from model_card_gen.analyze import get_analysis
        >>> get_analysis(model_path='compas/model',
                         data='compas/eval.tfrecord',
                         eval_config='compas/eval_config.proto')
    """
    analyzers = get_analyzers(model_path, eval_config, datasets)
    return [analyzer.run_analysis() for analyzer in analyzers]
