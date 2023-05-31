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
        "value has on a model's prediction. This is impact is measured by each "
        "feature values estimated Shapely value. Each plot represents an "
        "individual prediction. ",
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
    "Virtical Axis": 
        "Feature value for single prediction",
    "Expected Results":
        "The highest absolute Shapley value contributes the most to the "
        "model's prediction.",
}
