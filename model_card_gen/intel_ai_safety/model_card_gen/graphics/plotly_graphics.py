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

import plotly.express as px
from plotly import graph_objs as go
from plotly.io import to_html, templates
import pandas as pd
from typing import Optional, Union, Text, Sequence, Dict
from .plotly_utils import *

_COLORS = templates["plotly"].layout.colorway

def visualize_bool_values(trace_per_graph, graph_count):
    enum_range = range(0, trace_per_graph * graph_count, trace_per_graph)
    for left_fill, right_fill in zip(enum_range, enum_range[::-1]):
        yield (([False] * left_fill) + ([True] * trace_per_graph) + ([False] * right_fill))

class _PlotlyGraph:
    """Model Card Plotly graph"""

    # Attribute used to render graph
    eval_result_keys: Optional[Sequence[Text]] = None
    data: Optional[pd.DataFrame] = None
    x_name: Optional[Text] = None
    y_name: Optional[Text] = None
    title: Optional[Text] = None
    name: Optional[Text] = None
    labels: Optional[Dict] = None
    color: Optional[Text] = None
    eval_result_keys: Optional[Sequence[Text]] = None
    class_name: Optional[Text] = None
    metrics: Sequence[Text] = None

    # Graph generated from the data
    figure: Optional[go.Figure] = None
    base64str: Optional[Text] = None
    html_content: Optional[Text] = None

    @classmethod
    def generate_figure(cls, data):
        self = cls()
        # self.kwargs = kwargs
        self.data = self.validate_df(data)
        self.figure = self.create_fig(self.data)
        self.html_content = to_html(self.figure, include_plotlyjs="require", full_html=False)
        return self

    def validate_df(self, *args):
        pass

    def create_fig(self, data):
        pass


class OverallPerformanceAtThreshold(_PlotlyGraph):
    x_name: Optional[Text] = "threshold"
    y_name: Optional[Text] = "Overall"
    title: Optional[Text] = "Overall Accuracy/Precision/Recall/F1"
    labels = {"_value": "Accuracy/Precision/Recall/F1", "variable": ""}
    metric_names = ["accuracy", "precision", "recall", "f1"]

    def validate_df(self, df):
        assert (df.values.any()), "No values in DataFrame"
        assert (df.get('threshold') is not None), "No column named 'threshold'"
        assert (df['threshold'].values.any()), "Column named 'threshold' contains no values"
        assert (df['threshold'].dtype == 'float'), "Column named 'threshold' is not dtype float"
        for metric in self.metric_names:
            assert (df.get(metric) is not None), f"No column named '{metric}'"
            assert (df[metric].values.any()), f"Column named '{metric}' contains no values"
        if 'group' not in df:
            df['group'] = ['Overall'] * len(df)
        assert ('Overall' in df['group'].unique()), "Column named 'group' does not contain 'Overall' value"
        return df

    def create_fig(self, df):
        df = df[df["group"] == self.y_name]
        df = df[[self.x_name] + self.metric_names]
        fig = px.line(
            df.astype('float64'),
            x=self.x_name,
            y=self.metric_names,
            labels=self.labels,
            title=self.title,
        )
        fig.update_traces(mode="lines", hovertemplate=None)
        fig.update_layout(hovermode="x unified")
        return fig


class DataStatsGraphs(_PlotlyGraph):

    def validate_df(self, df):
        assert (df.values.any()), "No values in DataFrame"
        return df

    def create_fig(self, df):
        # Allows graph cycle through colorway
        if "color_index" in self.kwargs and isinstance(self.kwargs["color_index"], int):
            self.color = _COLORS[self.kwargs["color_index"] % len(_COLORS)]
        else:
            self.color = _COLORS[0]
        fig = go.Figure()
        ds = df.dataset.unique()[0]
        for feature in list(df.feature.unique()):
            vis = True if feature == list(df.feature.unique())[0] else False
            trace = go.Bar(
                x=df.loc[df.feature.isin([feature])].counts,
                y=df.loc[df.feature.isin([feature])].bins,
                name="{ds}: {feature}".format(ds=ds, feature=feature),
                orientation="h",
                visible=vis,
                marker=dict(color=self.color),
            )
            fig.add_trace(trace)

        def create_layout_button(feature, df):
            return dict(
                label=feature,
                method="restyle",
                args=[
                    {
                        "visible": [feat == feature for feat in list(df.feature.unique())],
                        "title": feature,
                        "showlegend": True,
                    }
                ],
            )

        fig.update_layout(
            updatemenus=[
                go.layout.Updatemenu(
                    active=0,
                    buttons=[create_layout_button(feature, df) for feature in list(df.feature.unique())],
                    x=0,
                    xanchor="left",
                    y=1.22,
                    yanchor="top",
                )
            ]
        )
        return fig


class ConfusionMatrixAtThresholdsGraphs(_PlotlyGraph):
    x_name: Optional[Text] = "threshold"
    y_name: Optional[Text] = "value"
    figure: Optional[go.Figure] = None

    def validate_df(self, df):
        assert (df.values.any()), "No values in DataFrame"
        assert (df.get('threshold') is not None), "No column named 'threshold'"
        assert (df['threshold'].values.any()), "Column named 'threshold' contains no values"
        assert (df['threshold'].dtype == 'float'), "Column named 'threshold' is not dtype float"
        if 'group' not in df:
            df['group'] = ['Overall'] * len(df)
        assert ('Overall' in df['group'].unique()), "Column named 'group' does not contain 'Overall' value"
        if 'feature' not in df:
            df['feature'] = ['Overall'] * len(df)
        assert ('Overall' in df['feature'].unique()), "Column named 'feature' does not contain 'Overall' value"
        assert (df.columns.drop(['feature', 'group', 'threshold']).any()), "No columns for metric names"
        return df

    def create_fig(self, df):
        metrics = df.columns.drop(['feature', 'group', 'threshold'])
        dfs = df.groupby("group")
        fig = go.Figure()
        for metric in metrics:
            for group, df in dfs:
                vis = True if metric == metrics[0] else False
                feature = df.feature.iloc[0]
                if feature == "Overall":
                    title = feature
                else:
                    title = f"{feature}={group}".replace("_", " ").title()
                trace = go.Scatter(x=df[self.x_name], y=df[metric], mode="lines", name=title, visible=vis)
                fig.add_trace(trace)

        def create_layout_button(metric, viz_arg):
            return dict(
                label=str(metric).title(),
                method="update",
                args=[{"visible": viz_arg}, {"title": f"{metric} at {self.x_name}".title()}],
            )

        metric_bools = visualize_bool_values(len(dfs), len(metrics))
        fig.update_layout(
            xaxis_title=self.x_name.title(),
            updatemenus=[
                go.layout.Updatemenu(
                    active=0,
                    buttons=[create_layout_button(metric, viz_arg) for metric, viz_arg in zip(metrics, metric_bools)],
                    x=0,
                    xanchor="left",
                    y=1.15,
                    yanchor="top",
                )
            ],
        )

        fig.update_traces(mode="lines", hovertemplate=None)
        fig.update_layout(hovermode="x unified")
        return fig


class SlicingMetricGraphs(_PlotlyGraph):
    x_name: Optional[Text] = "threshold"
    variables: Optional[Text] = "group"

    def validate_df(self, df):
        assert (df.values.any()), "No values in DataFrame"
        assert (df.get('feature') is not None), "No column named 'feature'"
        assert (df.get('group') is not None), "No column named 'group'"
        assert (df['feature'].values.any()), "Column named 'feature' contains no values"
        assert (df['group'].values.any()), "Column named 'group' contains no values"
        assert (df.columns.drop(['feature', 'group']).any()), "No columns for metric names"
        assert all(not df_group.empty for _, df_group in df.groupby('group')), "Not able to groupby 'group' column"
        return df

    def create_fig(self, df):
        dfs = df.groupby("group")
        fig = go.Figure()
        metrics = sorted(df.columns.drop(['feature', 'group']))
        for metric in metrics:
            for group, df in dfs:
                vis = True if metric == metrics[0] else False
                feature = df.feature.iloc[0]
                if feature == group:
                    title = feature
                else:
                    title = f"{df.feature.iloc[0]}={group}".replace("_", " ").title()
                trace = go.Bar(
                    x=df[metric],
                    y=df["group"].astype(str),
                    orientation="h",
                    name=title,
                    hovertemplate="%{y}: %{x}<extra></extra>",
                    visible=vis,
                )
                fig.add_trace(trace)

        def create_layout_button(metric, viz_arg):
            return dict(label=metric.title(), method="update", args=[{"visible": viz_arg}])

        metric_bools = visualize_bool_values(len(dfs), len(metrics))
        fig.update_layout(
            updatemenus=[
                go.layout.Updatemenu(
                    active=0,
                    buttons=[
                        create_layout_button(metric, viz_arg) for metric, viz_arg in zip(metrics, metric_bools)
                    ],
                    x=0,
                    xanchor="left",
                    y=1.15,
                    yanchor="top",
                )
            ]
        )

        return fig
