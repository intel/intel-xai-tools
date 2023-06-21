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

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Union


def shap_waterwall_plot(expected_value: float,
                        shap_values: np.ndarray,
                        feature_values: Union[pd.Series, np.ndarray],
                        columns: Union[pd.Index, np.array],
                        y_true=None,
                        y_pred=None,
                        sort=True,):
    """Builds plotly figure dipicting SHAP waterfall plot.

    Args:
        expected_value (float) : expected value or average shap value of
            explaination model
        shap_values (numpy.array) : array containing shap values for data point
        feature_values (pandas.Series or numpy.array) : array containing
            feature values for data point
        columns (pd.Index or numpy.array) : array containing feature names for
            dataset

    Kwargs:
        y_true (int or float) : number representing to true value for data
            point
        y_pred (int or float) : number representing to predicted value for
            data point
        sort (boolean) : boolean representing whether SHAP values should be
            sorted by impact score on plot
    Returns:
        plotly.graph_objects.Figure
    """
    if hasattr(feature_values, "values"):
        feature_values = feature_values.values
    fig = go.Figure(layout=go.Layout(hovermode='y unified'))
    pred = expected_value + shap_values.sum()
    if sort:
        ind = abs(shap_values).argsort()
        shap_values = np.take_along_axis(shap_values, ind, axis=0)
        feature_values = np.take_along_axis(feature_values, ind, axis=0)
        columns = np.take_along_axis(np.array(columns), ind, axis=0)
    impact_ranks = np.array(-abs(shap_values)).argsort().argsort() + 1
    xs = list(shap_values)
    ys = ["{}={}".format(c, v) for v, c in zip(feature_values, columns)]
    text = ['{:.3f}'.format(v) for v in shap_values]
    total = "Total SHAP Value"

    hovertemplate = "<br>".join([
        "<b>SHAP value</b>: %{x}",
        "<b>Feature value</b>: %{y}",
        "<b>Impact Rank<b>: %{customdata}"
        "<extra></extra>",
    ])

    fig.add_trace(go.Waterfall(
        x=xs,
        y=ys,
        text=text,
        measure=["relative"] * len(shap_values),
        base=expected_value,
        orientation="h",
        customdata=impact_ranks,
        decreasing={"marker": {"color": "#6d7da8"}},
        increasing={"marker": {"color": "#e88080 "}},
        hovertemplate=hovertemplate,
        showlegend=False,
    ))

    fig.add_trace(go.Waterfall(
        base=expected_value,
        orientation="h",
        text=['{:.3f}'.format(shap_values.sum())],
        measure=["absolute"],
        x=[shap_values.sum()],
        y=[total],
        showlegend=False,
        hovertemplate="Total SHAP value = <br>"
        "Base value + sum of all SHAP values<extra></extra>",
        totals={"marker": {"color": "rgba(115, 147, 179, 0)", "line": {
            "color": "#c7c7c7", "width": 2}}},
    ))

    fig.update_layout(title="SHAP Values")

    ys = [total, ys[0]]
    fig.add_trace(go.Scatter(
        x=[expected_value, expected_value],
        y=ys,
        mode="lines+text",
        textposition="bottom center",
        showlegend=False,
        line=dict(color="#c7c7c7", width=2, dash='dot'),
        hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=[pred, pred],
        y=ys,
        mode="lines+text",
        textposition="bottom center",
        showlegend=False,
        line=dict(color="#c7c7c7", width=2, dash='dot'),
        hoverinfo="skip",
    ))

    if y_pred:
        fig.add_trace(go.Scatter(
            x=[y_pred, y_pred],
            y=ys,
            mode="lines+text",
            textposition="bottom center",
            showlegend=False,
            line=dict(color="#c7c7c7", width=2, dash='dot'),
            hoverinfo="skip",
        ))

    if y_true is not None:
        fig.add_annotation(
            text=r"Ground truth y={}".format(y_true),
            y=1.17,
            xref='paper',
            yref='paper',
            showarrow=False,
            align='left'
        )
    fig.add_annotation(
        x=pred,
        y=total,
        text="Predicted Value={:.3f}".format(pred),
        showarrow=False,
        align="center",
        yshift=25,
    )

    fig.add_annotation(
        x=expected_value,
        y=ys[1],
        text="Base Value={:.3f}".format(expected_value),
        showarrow=False,
        align="center",
        yshift=-25,
    )
    return fig
