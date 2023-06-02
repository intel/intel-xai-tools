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
import os
import pkgutil
import json
from model_card_gen.model_card_gen import ModelCardGen
from model_card_gen.validation import (_LATEST_SCHEMA_VERSION,
                                       _SCHEMA_FILE_NAME,
                                       _find_json_schema,
                                       validate_json_schema)
PACKAGE = 'model_card_gen'
JSON_FILES = ['docs/examples/json/model_card_example.json', 'docs/examples/json/model_card_compas.json']
MODEL_CARD_STRS = [pkgutil.get_data(PACKAGE, json_file) for json_file in JSON_FILES]
MODEL_CARD_JSONS = [json.loads(json_str) for json_str in MODEL_CARD_STRS]


@pytest.mark.common
@pytest.mark.parametrize("test_json", MODEL_CARD_JSONS)
def test_init(test_json):
    """Test ModelCardGen initialization
    """
    mcg = ModelCardGen(data_sets={'test': ''}, model_card=test_json)
    assert mcg.model_card


@pytest.mark.common
@pytest.mark.parametrize("test_json", MODEL_CARD_JSONS)
def test_read_json(test_json):
    """Test ModelCardGen._read_json method
    """
    mcg = ModelCardGen(data_sets={'test': ''}, model_card=test_json)
    assert mcg.model_card == ModelCardGen._read_json(test_json)


@pytest.mark.common
@pytest.mark.parametrize("test_json", MODEL_CARD_JSONS)
def test_validate_json(test_json):
    """Test JSON validates
    """
    assert validate_json_schema(test_json) == _find_json_schema()


@pytest.mark.common
@pytest.mark.parametrize("test_json", MODEL_CARD_JSONS)
def test_schemas(test_json):
    """Test JSON schema loads
    """
    schema_file = os.path.join('schema', 'v' + _LATEST_SCHEMA_VERSION, _SCHEMA_FILE_NAME)
    json_file = pkgutil.get_data('model_card_gen', schema_file)
    schema = json.loads(json_file)
    assert schema == _find_json_schema(_LATEST_SCHEMA_VERSION)
