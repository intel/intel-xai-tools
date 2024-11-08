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

# Add Plotly visualizations to Model Card
from intel_ai_safety.model_card_gen.model_card import ModelCard, Graphic, Dataset, GraphicsCollection
from .plotly_graphics import *

# Typing
import pandas as pd
from typing import Sequence, Text, Tuple, Union, Optional, List
from base64 import b64encode

OVERVIEW_GRAPHS = [OverallPerformanceAtThreshold]
THRESHOLD_GRAPHS = [MetricsAtThresholdsGraphs]
SLICING_METRIC_GRAPHS = [SlicingMetricGraphs]
DATASTAT_GRAPHS = [DataStatsGraphs]


def add_overview_graphs(model_card: ModelCard, df: pd.DataFrame, static: bool) -> None:
    if "dataset" in df:
        dfs = df.groupby("dataset")
        for dataset_name, df in dfs:
            if "label" in df:
                label_dfs = df.groupby("label")
                for label_name, df_label in label_dfs:
                    df_label.drop("label", axis=1, inplace=True)
                    graphs = [graph.generate_figure(df_label) for graph in OVERVIEW_GRAPHS]

                    if static:
                        model_card.model_details.graphics.collection.extend(
                            [
                                Graphic(
                                    name=graph.title + f" ({dataset_name} - Label : {label_name})",
                                    image=b64encode(graph.figure.to_image(format="png")).decode(),
                                )
                                for graph in graphs
                            ]
                        )
                    else:
                        model_card.model_details.graphics.collection.extend(
                            [
                                Graphic(
                                    name=graph.title + f" ({dataset_name} - Label : {label_name})",
                                    html=graph.html_content,
                                )
                                for graph in graphs
                            ]
                        )
            else:
                graphs = [graph.generate_figure(df) for graph in OVERVIEW_GRAPHS]

                if static:
                    model_card.model_details.graphics.collection.extend(
                        [
                            Graphic(
                                name=graph.title + f" ({dataset_name})",
                                image=b64encode(graph.figure.to_image(format="png")).decode(),
                            )
                            for graph in graphs
                        ]
                    )
                else:
                    model_card.model_details.graphics.collection.extend(
                        [Graphic(name=graph.title + f" ({dataset_name})", html=graph.html_content) for graph in graphs]
                    )
    else:
        if "label" in df:
            label_dfs = df.groupby("label")
            for label_name, df_label in label_dfs:
                df_label.drop("label", axis=1, inplace=True)
                graphs = [graph.generate_figure(df_label) for graph in OVERVIEW_GRAPHS]

                if static:
                    model_card.model_details.graphics.collection.extend(
                        [
                            Graphic(
                                name=graph.title + f" - Label : ({label_name})",
                                image=b64encode(graph.figure.to_image(format="png")).decode(),
                            )
                            for graph in graphs
                        ]
                    )
                else:
                    model_card.model_details.graphics.collection.extend(
                        [
                            Graphic(name=graph.title + f" - Label : ({label_name})", html=graph.html_content)
                            for graph in graphs
                        ]
                    )
        else:
            graphs = [graph.generate_figure(df) for graph in OVERVIEW_GRAPHS]
            # Add graphs to model card
            if static:
                model_card.model_details.graphics.collection.extend(
                    [
                        Graphic(name=graph.title, image=b64encode(graph.figure.to_image(format="png")).decode())
                        for graph in graphs
                    ]
                )
            else:
                model_card.model_details.graphics.collection.extend(
                    [Graphic(name=graph.title, html=graph.html_content) for graph in graphs]
                )


def add_dataset_feature_statistics_plots(
    model_card: ModelCard, data_set_names: List[Text], dfs: List[pd.DataFrame]
) -> None:
    """Adds Dataset objects to model card with all graphs in
    DATASTAT_GRAPHS
    """
    named_datasets = zip(data_set_names, dfs)
    for i, (name, df) in enumerate(named_datasets):
        graphs = [graph.generate_figure(name, df, color_index=i) for graph in DATASTAT_GRAPHS]
        model_card.model_parameters.data.append(
            Dataset(
                name=name.title(),
                graphics=GraphicsCollection(
                    collection=[Graphic(name=graph.title, html=graph.html_content) for graph in graphs]
                ),
            )
        )


def add_eval_result_slicing_metrics(model_card: ModelCard, df: pd.DataFrame, static: bool) -> None:
    """Adds plots for every graph in SLICING_METRIC_GRAPHS
    and every metric in eval_result.slicing_metrics.

    Args:
        model_card: The model card object.
        eval_result: A `tfma.EvalResult`.
        static: A boolean flag to determine if the plots should be static (True) or interactive (False).
    """
    if "dataset" in df:
        dfs = df.groupby("dataset")
        for dataset_name, df in dfs:
            if "label" in df:
                label_dfs = df.groupby("label")
                for label_name, df_label in label_dfs:
                    df_label.drop("label", axis=1, inplace=True)
                    graphs = [graph.generate_figure(df_label) for graph in SLICING_METRIC_GRAPHS]
                    if static:
                        model_card.quantitative_analysis.graphics.collection.extend(
                            [
                                Graphic(
                                    name=graph.title + f" ({dataset_name}) - Label : {label_name})",
                                    image=b64encode(graph.figure.to_image(format="png")).decode(),
                                )
                                for graph in graphs
                            ]
                        )
                    else:
                        model_card.quantitative_analysis.graphics.collection.extend(
                            [
                                Graphic(
                                    name=graph.title + f" ({dataset_name}) - Label : {label_name})",
                                    html=graph.html_content,
                                )
                                for graph in graphs
                            ]
                        )
            else:
                graphs = [graph.generate_figure(df) for graph in SLICING_METRIC_GRAPHS]
                if static:
                    model_card.quantitative_analysis.graphics.collection.extend(
                        [
                            Graphic(
                                name=graph.title + f" ({dataset_name})",
                                image=b64encode(graph.figure.to_image(format="png")).decode(),
                            )
                            for graph in graphs
                        ]
                    )
                else:
                    model_card.quantitative_analysis.graphics.collection.extend(
                        [Graphic(name=graph.title + f" ({dataset_name})", html=graph.html_content) for graph in graphs]
                    )
    else:
        if "label" in df:
            label_dfs = df.groupby("label")
            for label_name, df_label in label_dfs:
                df_label.drop("label", axis=1, inplace=True)
                graphs = [graph.generate_figure(df_label) for graph in SLICING_METRIC_GRAPHS]
                if static:
                    model_card.quantitative_analysis.graphics.collection.extend(
                        [
                            Graphic(
                                name=graph.title + f" - Label : ({label_name})",
                                image=b64encode(graph.figure.to_image(format="png")).decode(),
                            )
                            for graph in graphs
                        ]
                    )
                else:
                    model_card.quantitative_analysis.graphics.collection.extend(
                        [
                            Graphic(name=graph.title + f" - Label : ({label_name})", html=graph.html_content)
                            for graph in graphs
                        ]
                    )

        else:
            graphs = [graph.generate_figure(df) for graph in SLICING_METRIC_GRAPHS]
            if static:
                model_card.quantitative_analysis.graphics.collection.extend(
                    [
                        Graphic(name=graph.title, image=b64encode(graph.figure.to_image(format="png")).decode())
                        for graph in graphs
                    ]
                )
            else:
                model_card.quantitative_analysis.graphics.collection.extend(
                    [Graphic(name=graph.title, html=graph.html_content) for graph in graphs]
                )


def add_eval_result_plots(model_card: ModelCard, df: pd.DataFrame, static: bool) -> None:
    """Add visualizations for every plot in eval_result.plots.

    This function generates plots encoded as html text
    strings, and appends them to
    model_card.quantitative_analysis.graphics.collection.

    Args:
        model_card: The model card object.
        eval_result: A `tfma.EvalResult`.
        static: A boolean flag to determine if the plots should be static (True) or interactive (False).
    """
    if "dataset" in df:
        dfs = df.groupby("dataset")
        for dataset_name, df in dfs:
            if "label" in df:
                label_dfs = df.groupby("label")
                for label_name, df_label in label_dfs:
                    df_label.drop("label", axis=1, inplace=True)
                    graphs = [graph.generate_figure(df_label) for graph in THRESHOLD_GRAPHS]
                    if static:
                        model_card.quantitative_analysis.graphics.collection.extend(
                            [
                                Graphic(
                                    name=graph.title + f" ({dataset_name}) - Label : {label_name})",
                                    image=b64encode(graph.figure.to_image(format="png")).decode(),
                                )
                                for graph in graphs
                            ]
                        )
                    else:
                        model_card.quantitative_analysis.graphics.collection.extend(
                            [
                                Graphic(
                                    name=graph.title + f" ({dataset_name}) - Label : {label_name})",
                                    html=graph.html_content,
                                )
                                for graph in graphs
                            ]
                        )
            else:
                graphs = [graph.generate_figure(df) for graph in THRESHOLD_GRAPHS]
                if static:
                    model_card.quantitative_analysis.graphics.collection.extend(
                        [
                            Graphic(
                                name=graph.title + f" ({dataset_name})",
                                image=b64encode(graph.figure.to_image(format="png")).decode(),
                            )
                            for graph in graphs
                        ]
                    )
                else:
                    model_card.quantitative_analysis.graphics.collection.extend(
                        [Graphic(name=graph.title + f" ({dataset_name})", html=graph.html_content) for graph in graphs]
                    )
    else:
        if "label" in df:
            label_dfs = df.groupby("label")
            for label_name, df_label in label_dfs:
                df_label.drop("label", axis=1, inplace=True)
                graphs = [graph.generate_figure(df_label) for graph in THRESHOLD_GRAPHS]
                if static:
                    model_card.quantitative_analysis.graphics.collection.extend(
                        [
                            Graphic(
                                name=graph.title + f" - Label : ({label_name})",
                                image=b64encode(graph.figure.to_image(format="png")).decode(),
                            )
                            for graph in graphs
                        ]
                    )
                else:
                    model_card.quantitative_analysis.graphics.collection.extend(
                        [
                            Graphic(name=graph.title + f" - Label : ({label_name})", html=graph.html_content)
                            for graph in graphs
                        ]
                    )

        else:
            graphs = [graph.generate_figure(df) for graph in THRESHOLD_GRAPHS]
            if static:
                model_card.quantitative_analysis.graphics.collection.extend(
                    [
                        Graphic(name=graph.title, image=b64encode(graph.figure.to_image(format="png")).decode())
                        for graph in graphs
                    ]
                )
            else:
                model_card.quantitative_analysis.graphics.collection.extend(
                    [Graphic(name=graph.title, html=graph.html_content) for graph in graphs]
                )


def get_slice_key(slice_key: Union[Tuple[()], Tuple[Tuple[Text, Union[Text, int, float]], ...]]) -> Tuple[Text, Text]:
    """Returns a tuple of joined keys and values accross slice_key"""

    if not len(slice_key):
        return ("Overall", "Overall")

    keys, values = zip(*slice_key)

    return (", ".join(["{}".format(key) for key in keys]), ", ".join(["{}".format(value) for value in values]))
