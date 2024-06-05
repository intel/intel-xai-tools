#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Intel Corporation
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

import shap
from typing import List
from intel_ai_safety.explainer.context.agnostic.attributions_explainer import AttributionsExplainer


class LLMExplainer(AttributionsExplainer):
    """
    Approximate an extension of shap values, known as Owen values, for generative and classification LLMs from Hugging Face.

    Args:
      model (HF Automodel or pipeline): model to be interpreted. If no tokenizer is supplied than model is assumed to be
      a pipeline
      tokenizer (HF Tokenizer): Toke

    reference:
    https://shap-lrjball.readthedocs.io/en/latest/generated/shap.PartitionExplainer.html
    """

    def __init__(self, model, tokenizer=None) -> None:

        super().__init__(self)
        self.shap_values = None
        self.explainer = shap.Explainer(model, tokenizer)

    def run_explainer(self, target_text: str, max_evals: int = 64) -> None:
        """
        Execute the partition explanation on the target_text.

        Args:
          target_text (numpy.ndarray): 1-d numpy array of strings holding the text examples
          max_evals (int): number of evaluations used in the shap estimation. The higher the number result
          in a better the estimation. Defaults to 64.

        Returns:
          None
        """
        self.shap_values = self.explainer(target_text, max_evals=max_evals)

    def visualize(self):
        """
        Display the force plot of the of the target example(s)
        """
        shap.text_plot(self.shap_values)


def llm_explainer(model, target_text, tokenizer=None, max_evals=64):
    """
    Instantiates LLMExplainer for text classification (currently with Pipeline class only)
    or generation (Pipeline and Auto classes) explanation and executes run_explainer()

    Args:
      model (function): the model
      target_text (numpy.ndarray): 1-d numpy array of strings holding the text examples
      tokenizer (string or callable): tokenizer associated with model
      max_evals (int):

    Reference:
      https://shap-lrjball.readthedocs.io/en/latest/generated/shap.PartitionExplainer.html

    Returns:
      LLMExplainer
    """
    llme = LLMExplainer(model, tokenizer)
    llme.run_explainer(target_text, max_evals=max_evals)
    return llme
