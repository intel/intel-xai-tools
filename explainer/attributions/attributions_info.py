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

force_plot_info_panel = {
    "Plot Description":
        "This graph shows the attributed impact each feature"
        "value has on a model's prediction. This impact is measured by "
        "each feature value's estimated Shapely value. Each plot represents an"
        " individual prediction.",
    "Metric Description":
        "Shapley values is a concept borrowed from coalitional game theory. "
        "In its original application, this value can tell us how much "
        "payoff each player can reasonably expect given their contribution "
        "to the coalition. Applied to machine learning, each feature value "
        "for an instance is a player in a game and the prediction is the "
        "payout. Thus, the Shapley value for a prediction measures the "
        "relative the contribution each feature value had to the prediction.",
    "Plot Colors":
        "Red represents a positive impact. Blue represents a negative impact.",
    "Horizontal Axis":
        "Estimated Shapley value (impact on model prediction)",
    "Vertical Axis":
        "Feature and feature values for single prediction",
    "Expected Results":
        "The highest absolute Shapley value contributes the most to the "
        "model's prediction.",
}

shap_widget_info_panel = {
    "Error Analysis":
        "Allows the user to filter data points based on their error type.",
    "Impact Analysis":
        "Allows users to filter data points based on their associated SHAP "
        "impact score for top important features",
    "Feature Analysis":
        "Allows users to filter data points feature values.",
    "Base Value":
        "Refers to the expected or average SHAP value of the"
        "explaination model.",
    "Predicted Value":
        "Refers the estimated output score of the model calculated"
        "by base value plus the sum of SHAP values across all features."
    }
shap_widget_info_panel.update(force_plot_info_panel)
