#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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

### libraries to support tests ###
import pytest

### library to be tested ###
from intel_ai_safety.explainer.attributions import hf_attributions as attributions


@pytest.mark.parametrize("auto_model", ["causal_lm", "seq2seq_lm"])
def test_llm_explainer_auto_model(auto_model, request):
    """
    Test LLMExplainer works with AutoModelForCausalLM and AutoModelForSeq2SeqLM
    """
    model, tokenizer, text = request.getfixturevalue(auto_model)
    llme = attributions.LLMExplainer(model, tokenizer)
    llme.run_explainer(text)
    assert isinstance(llme, attributions.LLMExplainer)
    llme.visualize()


def test_llm_explainer_pipeline(classification_pipeline):
    """
    Test LLMExplainer works with classification pipeline
    """
    classifier_pl, text = classification_pipeline
    llme = attributions.LLMExplainer(classifier_pl)
    llme.run_explainer(text)
    assert isinstance(llme, attributions.LLMExplainer)
    llme.visualize()
