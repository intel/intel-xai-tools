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

import pytest
import tempfile
import tensorflow as tf
from model_card_gen.tests.model import build_and_train_model, train_tf_file, validate_tf_file
from model_card_gen.model_card_gen import ModelCardGen
import tensorflow_model_analysis as tfma
from google.protobuf import text_format


def test_end_to_end():
    """ Build a model card from a trained model
    """
    tfma_export_dir = build_and_train_model()
    _model_path = tfma_export_dir
    _data_paths = {'eval': validate_tf_file,
                'train': train_tf_file}

    eval_config = text_format.Parse("""
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

    mcg = ModelCardGen.generate(_data_paths, _model_path, eval_config)
    assert mcg
