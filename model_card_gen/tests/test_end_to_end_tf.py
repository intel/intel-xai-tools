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
import shutil

import pytest
import tensorflow_model_analysis as tfma
from google.protobuf import text_format

from model_card_gen.model_card_gen import ModelCardGen


@pytest.mark.tensorflow
class TestMCGForTensorFlow:

    @classmethod
    def setup_class(cls):
        # Do TensorFlow specific imports within the marked class so that we don't get errors when TF isn't installed
        from model_card_gen.datasets import TensorflowDataset
        from model_card_gen.tests.tf_model import build_and_train_model, train_tf_file, validate_tf_file

        cls._model_path = build_and_train_model()
        cls._data_sets = {'eval': TensorflowDataset(dataset_path=validate_tf_file),
                          'train': TensorflowDataset(dataset_path=train_tf_file)}

        cls._eval_config = text_format.Parse("""
            model_specs {
                signature_name: "eval"
            }

            metrics_specs {
                metrics { class_name: "BinaryAccuracy" }
                metrics { class_name: "Precision" }
                metrics { class_name: "Recall" }
                metrics { class_name: "ConfusionMatrixPlot" }
                metrics { class_name: "FairnessIndicators" }
            }

            slicing_specs {}  # overall slice
            slicing_specs {
                feature_keys: ["gender"]
            }
            """, tfma.EvalConfig())

    @classmethod
    def teardown_class(cls):
        if os.path.exists(cls._model_path):
            shutil.rmtree(cls._model_path)

    def test_end_to_end(self):
        """ Build a model card from a trained model
        """

        mcg = ModelCardGen.generate(data_sets=self._data_sets, model_path=self._model_path,
                                    eval_config=self._eval_config)

        assert mcg
