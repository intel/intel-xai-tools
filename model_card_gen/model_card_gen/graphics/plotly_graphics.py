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
from plotly.subplots import make_subplots
from plotly.io import to_html, templates
import pandas as pd
from typing import Optional, Union, Text, Sequence, Dict
from .plotly_utils import *

_COLORS = templates["plotly"].layout.colorway

class _PlotlyGraph():
    """Model Card Plotly graph"""

    # Attribute used to render graph
    eval_result_keys: Optional[Sequence[Text]] = None
    data : Optional[pd.DataFrame] = None
    x_name: Optional[Text] = None
    y_name: Optional[Text] = None
    title: Optional[Text] = None
    name: Optional[Text] = None
    labels: Optional[Dict] = None
    color: Optional[Text] = None
    eval_result_keys: Optional[Sequence[Text]] = None
    class_name : Optional[Text] = None
    metrics: Sequence[Text] = None

    # Graph generated from the data
    figure: Optional[go.Figure] = None
    base64str: Optional[Text] = None
    html_content: Optional[Text] = None

    @classmethod
    def generate_figure(cls, *args, **kwargs):
        self = cls()
        self.kwargs = kwargs
        self.metrics =  self.create_metrics(*args)
        self.data = self.create_df(*args)
        self.figure = self.create_fig(self.data)
        self.html_content = to_html(self.figure, include_plotlyjs='require', full_html=False)
        return self

    def create_metrics(self, *args):
        pass

    def create_df(self, *args):
        pass

    def create_fig(self, data):
        pass

class OverallPerformanceAtThreshold(_PlotlyGraph):
    eval_result_keys: Optional[Sequence[Text]] = ['confusionMatrixAtThresholds', 'matrices']
    x_name: Optional[Text] = 'threshold'
    y_name: Optional[Text] = 'Overall'
    title: Optional[Text] = "Overall Accuracy/Precision/Recall/F1"
    labels = {"_value": "Accuracy/Precision/Recall/F1", "variable": ""}

    def create_df(self, plots):
        df = plots_to_df(plots, self.eval_result_keys)
        num = df.get('truePositives', 0) + df.get('trueNegatives', 0)
        dom = (df.get('truePositives', 0) + df.get('falsePositives', 0) + 
               df.get('falseNegatives', 0) + df.get('trueNegatives', 0))
        df['accuracy'] = num / dom
        df['f1'] = 2 * (df.precision * df.recall) / (df.precision + df.recall)
        return df
    
    def create_fig(self, df):
        fig = px.line(self.data[self.data["group"] == self.y_name],
                      x=self.x_name,
                      y= ['accuracy', 'precision', 'recall', 'f1'],
                      labels=self.labels,
                      title = self.title)
        fig.update_traces(mode="lines", hovertemplate=None)
        fig.update_layout(hovermode="x unified")
        return fig

class DataStatsGraphs(_PlotlyGraph):
        
    def create_df(self, name, stats):
        data = pd.DataFrame()
        for dataset in stats.datasets:
            for feature in dataset.features:
                df = pd.DataFrame()
                if feature.HasField('num_stats') and feature.num_stats.histograms:
                    histogram = feature.num_stats.histograms[0]
                    # x-axis
                    df['counts'] = [int(bucket.sample_count) for bucket in histogram.buckets]
                    # y-axis
                    df['bins'] = [
                        f'{bucket.low_value:.2f}-{bucket.high_value:.2f}'
                        for bucket in histogram.buckets
                    ]
                    # button filter
                    df['feature'] = feature.name or feature.path.step[0]
                    # groupby feature
                    df['dataset'] = 'Dataset {name}'.format(name=name)
                    data = pd.concat([data, df])
                if feature.HasField('string_stats'):
                    rank_histogram = feature.string_stats.rank_histogram
                    df['counts'] = [int(bucket.sample_count) for bucket in rank_histogram.buckets]
                    df['bins'] = [bucket.label for bucket in rank_histogram.buckets]
                     # button filter
                    df['feature'] = feature.name or feature.path.step[0]
                    # groupby feature
                    df['dataset'] = 'Dataset {name}'.format(name=str(name).title())
                    data = pd.concat([data, df])
        return data

    def create_fig(self, df):
        # Allows graph cycle through colorway
        if 'color_index' in self.kwargs and isinstance(self.kwargs['color_index'], int):
            self.color = _COLORS[self.kwargs['color_index'] % len(_COLORS)]
        else:
            self.color = _COLORS[0]
        fig = go.Figure()
        ds = df.dataset.unique()[0]
        for feature in list(df.feature.unique()):
            vis = True if feature == list(df.feature.unique())[0] else False
            trace = go.Bar(
                x=df.loc[df.feature.isin([feature])].counts,
                y=df.loc[df.feature.isin([feature])].bins,
                name = '{ds}: {feature}'.format(ds=ds, feature=feature),
                orientation='h',
                visible=vis,
                marker=dict(color=self.color)
            )
            fig.add_trace(trace)

        def create_layout_button(feature, df):
            return dict(label = feature,
                        method = 'restyle',
                        args = [{'visible': [feat == feature for feat in list(df.feature.unique())],
                                 'title': feature,
                                 'showlegend': True}])

        fig.update_layout(
            updatemenus=[go.layout.Updatemenu(
                active = 0,
                buttons = [create_layout_button(feature, df)
                    for feature in list(df.feature.unique())],
                x = 0,
                xanchor = 'left',
                y = 1.22,
                yanchor = 'top',
                )
            ])
        return fig

class ConfusionMatrixAtThresholdsGraphs(_PlotlyGraph):
    eval_result_keys: Optional[Sequence[Text]] = ['confusionMatrixAtThresholds', 'matrices']
    x_name: Optional[Text] = 'threshold'
    y_name: Optional[Text] = 'value'

    figure: Optional[go.Figure] = None
        
    def create_df(self, plots):
        df = plots_to_df(plots, self.eval_result_keys)
        num = df.get('truePositives', 0) + df.get('trueNegatives', 0)
        dom = (df.get('truePositives', 0) + df.get('falsePositives', 0) + 
               df.get('falseNegatives', 0) + df.get('trueNegatives', 0))
        df['accuracy'] = num / dom
        df['f1'] = 2 * (df.precision * df.recall) / (df.precision + df.recall)
        return df
    
    def create_fig(self, df):
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        dfs = df.groupby("group")
        fig = go.Figure()
        for metric in metrics:
            for group, df in dfs:
                vis = True if metric == metrics[0] else False
                feature = df.feature.iloc[0]
                if feature == "Overall":
                    title = feature
                else:
                    title = f'{feature}={group}'.replace('_', " ").title()
                trace = go.Scatter(x=df[self.x_name],
                                y=df[metric],
                                mode='lines',
                                name=title,
                                visible=vis)
                fig.add_trace(trace)
                
        def create_layout_button(metric, viz_arg):
            return dict(label = str(metric).title(),
                        method = 'update',
                        args = [{'visible': viz_arg},
                                {'title': f'{metric} at {self.x_name}'.title()}])
        
        metric_bools = visualize_bool_values(len(dfs), len(metrics))
        fig.update_layout(
            xaxis_title=self.x_name.title(),
            updatemenus=[go.layout.Updatemenu(
                    active=0,
                    buttons=[create_layout_button(metric, viz_arg)
                                for metric, viz_arg in zip(metrics, metric_bools)],
                            x = 0,
                            xanchor = 'left',
                            y = 1.15,
                            yanchor = 'top')])

        fig.update_traces(mode="lines", hovertemplate=None)
        fig.update_layout(hovermode="x unified")
        return fig

class SlicingMetricGraphs(_PlotlyGraph):
    x_name: Optional[Text] = 'threshold'
    variables : Optional[Text] = 'group'
        
    def create_metrics(self, slicing_metrics):
        metrics = set()
        for slicing_metric in slicing_metrics:
            for output_name in slicing_metric[1]:
                for sub_key in slicing_metric[1][output_name]:
                    metrics.update(slicing_metric[1][output_name][sub_key].keys())
        return sorted(metrics)
        
    def create_df(self, slicing_metrics):
        return slicing_metric_to_df(slicing_metrics)
    
    def create_fig(self, df):
        dfs = df.groupby("group")
        fig = go.Figure()
        for metric in self.metrics:
            for group, df in dfs:
                vis = True if metric == self.metrics[0] else False
                feature = df.feature.iloc[0]
                if feature == "Overall":
                    title = feature
                else:
                    title = f'{df.feature.iloc[0]}={group}'.replace('_', " ").title()
                trace = go.Bar(x=df[metric],
                       y=df['group'].astype(str),
                       orientation='h',
                       name=title,
                       hovertemplate = '%{y}: %{x}<extra></extra>',
                       visible=vis)
                fig.add_trace(trace)

        def create_layout_button(metric, viz_arg):
            return dict(label = metric.title(),
                        method = 'update',
                        args = [{'visible': viz_arg}])
        
        metric_bools = visualize_bool_values(len(dfs), len(self.metrics))
        fig.update_layout(
            updatemenus=[go.layout.Updatemenu(
                    active=0,
                    buttons=[create_layout_button(metric, viz_arg)
                                for metric, viz_arg in zip(self.metrics, metric_bools)],
                    x = 0,
                    xanchor = 'left',
                    y = 1.15,
                    yanchor = 'top'
            )])

        return fig
