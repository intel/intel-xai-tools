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

from functools import reduce
import operator
import pandas as pd
import tensorflow_model_analysis as tfma
from typing import Optional, Union, Text, Sequence, Dict

def plot_results_parser(
    container,
    keys,
    feature_key_value=None):
    # Get feature and value
    if isinstance(container, tuple) and feature_key_value == None:
        if container[0] and container[0][0]:
            feature, value = container[0][0]
        else:
            feature, value = 'Overall', 'Overall'
        return plot_results_parser(container[1], keys, (feature, value))
    # Recursively retrieve value form empty string keys
    elif isinstance(container, dict) and list(container.keys()) == ['']:
        return plot_results_parser(container.get(''), keys, feature_key_value)
    elif isinstance(container, dict):
        try:
            feature, value = feature_key_value
            return (dict(item, **{'feature':feature, 'group': value})
                    for item in reduce(operator.getitem, keys, container))
        except (IndexError, KeyError):
            return None

def plots_to_df(plots, keys):
    return pd.DataFrame.from_records([result for plot in plots
        for result in plot_results_parser(plot, keys)])

def slicing_metric_to_df(
    slicing_metrics: Sequence[tfma.view.SlicedMetrics],
    output_name: Text = '',
    sub_key: Text = '') -> pd.DataFrame:
    records  = []
    for slicing_metric in slicing_metrics:
        if slicing_metric[0] and slicing_metric[0][0]:
            feature, value = slicing_metric[0][0]
        else:
            feature, value = 'Overall', 'Overall'
        record = {'feature': feature, 'group': value}
        for metric in slicing_metric[1][output_name]['']:
            for value_type in slicing_metric[1][output_name][sub_key][metric]:
                metric_value = slicing_metric[1][output_name][sub_key][metric][value_type]
                record[metric] = metric_value
        records.append(record)
    return pd.DataFrame(records)

def visualize_bool_values(trace_per_graph, graph_count):
    enum_range = range(0, trace_per_graph * graph_count, trace_per_graph)
    for left_fill, right_fill in zip(enum_range, enum_range[::-1]):
        yield (([False] * left_fill) + 
               ([True] * trace_per_graph) +
               ([False] * right_fill))

def show_nth_trace(n, trace_per_graph, graph_count):
    """ Return a generator of bools such that the nth (count stating at 1)
    of a group of size `trace_per_graph` of a collection of `graph_count`
    groups.

    Example:
        _show_nth_trace(2, 3, 2) returns iter([False, True, False, True])
    """
    if n > trace_per_graph:
        raise ValueError(f"Cannot show {n}th item of a group of size {trace_per_graph}")
    for i in range(trace_per_graph * graph_count):
        if i % trace_per_graph == (n - 1) and (n == 1 or i != 0):
            yield True
        else:
            yield False
