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

import os
import pytest

import tensorflow_model_analysis as tfma
from google.protobuf import text_format

from model_card_gen.model_card_gen import ModelCardGen


@pytest.mark.pytorch
class TestMCGForPyTorch:

    @classmethod
    def setup_class(cls):
        # Do PyTorch specific imports within the marked class so that we don't get errors when torch isn't installed
        from model_card_gen.datasets import PytorchDataset
        from model_card_gen.tests.torch_model import get_data, get_trained_model

        adult_dataset, feature_names = get_data()
        cls._model_path = get_trained_model(adult_dataset, feature_names)
        cls._data_sets = {'train': PytorchDataset(adult_dataset, feature_names=feature_names)}

        cls._eval_config = text_format.Parse("""
                model_specs {
                label_key: 'label'
                prediction_key: 'prediction'
                }
                metrics_specs {
                    metrics {class_name: "BinaryAccuracy"}
                    metrics {class_name: "AUC"}
                    metrics {class_name: "ConfusionMatrixPlot"}
                    metrics {
                    class_name: "FairnessIndicators"
                    }
                }
                slicing_specs {}
                slicing_specs {
                        feature_keys: 'sex_Female'
                }
                options {
                    include_default_metrics { value: false }
                }
                """, tfma.EvalConfig())

    @classmethod
    def teardown_class(cls):
        if os.path.exists(cls._model_path):
            os.remove(cls._model_path)

    def test_end_to_end(self):
        """ Build a pytorch model card from a trained model
        """
        mcg = ModelCardGen.generate(data_sets=self._data_sets, model_path=self._model_path,
                                    eval_config=self._eval_config)

        assert mcg
