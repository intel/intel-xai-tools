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

from typing import Sequence, Text, Tuple, Union, List
import tensorflow_model_analysis as tfma
import pandas as pd
from functools import reduce
import operator


def plot_results_parser(container, keys, feature_key_value=None):
    # Get feature and value
    if isinstance(container, tuple) and feature_key_value == None:
        if container[0] and container[0][0]:
            feature, value = container[0][0]
        else:
            feature, value = "Overall", "Overall"
        return plot_results_parser(container[1], keys, (feature, value))
    # Recursively retrieve value form empty string keys
    elif isinstance(container, dict) and list(container.keys()) == [""]:
        return plot_results_parser(container.get(""), keys, feature_key_value)
    elif isinstance(container, dict):
        try:
            feature, value = feature_key_value
            return (
                dict(item, **{"feature": feature, "group": value}) for item in reduce(operator.getitem, keys, container)
            )
        except (IndexError, KeyError):
            return None


def plots_to_df(plots, keys):
    return pd.DataFrame.from_records([result for plot in plots for result in plot_results_parser(plot, keys)])


def slicing_metric_to_df(
    slicing_metrics: Sequence[tfma.view.SlicedMetrics], output_name: Text = "", sub_key: Text = ""
) -> pd.DataFrame:
    records = []
    for slicing_metric in slicing_metrics:
        if slicing_metric[0] and slicing_metric[0][0]:
            feature, value = slicing_metric[0][0]
        else:
            feature, value = "Overall", "Overall"
        record = {"feature": feature, "group": value}
        for metric in slicing_metric[1][output_name][""]:
            for value_type in slicing_metric[1][output_name][sub_key][metric]:
                metric_value = slicing_metric[1][output_name][sub_key][metric][value_type]
                record[metric] = metric_value
        records.append(record)
    return pd.DataFrame(records)


def tfdv_to_df(name, stats):
        data = pd.DataFrame()
        for dataset in stats.datasets:
            for feature in dataset.features:
                df = pd.DataFrame()
                if feature.HasField("num_stats") and feature.num_stats.histograms:
                    histogram = feature.num_stats.histograms[0]
                    # x-axis
                    df["counts"] = [int(bucket.sample_count) for bucket in histogram.buckets]
                    # y-axis
                    df["bins"] = [f"{bucket.low_value:.2f}-{bucket.high_value:.2f}" for bucket in histogram.buckets]
                    # button filter
                    df["feature"] = feature.name or feature.path.step[0]
                    # groupby feature
                    df["dataset"] = "Dataset {name}".format(name=name)
                    data = pd.concat([data, df])
                if feature.HasField("string_stats"):
                    rank_histogram = feature.string_stats.rank_histogram
                    df["counts"] = [int(bucket.sample_count) for bucket in rank_histogram.buckets]
                    df["bins"] = [bucket.label for bucket in rank_histogram.buckets]
                    # button filter
                    df["feature"] = feature.name or feature.path.step[0]
                    # groupby feature
                    df["dataset"] = "Dataset {name}".format(name=str(name).title())
                    data = pd.concat([data, df])
        return data


def get_slice_key(slice_key: Union[Tuple[()], Tuple[Tuple[Text, Union[Text, int, float]], ...]]) -> Tuple[Text, Text]:
    """Returns a tuple of joined keys and values accross slice_key"""

    if not len(slice_key):
        return ("Overall", "Overall")

    keys, values = zip(*slice_key)

    return (", ".join(["{}".format(key) for key in keys]), ", ".join(["{}".format(value) for value in values]))
