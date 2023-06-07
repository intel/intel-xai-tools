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

import ipywidgets as widgets
from IPython.display import display
import pandas as pd
from pandas.api.types import is_float_dtype
import numpy as np
import plotly.graph_objects as go
from typing import List
from .plots import shap_waterwall_plot
from .attributions_info import shap_widget_info_panel
from explainer.utils.graphics.info import InfoPanel


class _FilterWidget:
    """Base class that builds a filter UI from pandas.DataFrame. Each field
    with float data type is given a slider to filter dataframe indices based
    on defined range. Other data types are given a select multiple dropdown
    to filter  dataframe indices based selected
    values.

    Args:
        df (pandas.DataFrame) : dataframe of filtering interface to be
            displayed
        max_values (int) : max number of values supported by interface

    Attributes:
        view (widgets.VBox) : widget class to display in Jupyter enviornment
        max_values (int) : max number of values supported by interface
    """
    def __init__(self, df, max_values=2000):
        self.max_values = max_values
        self._df = df[:self.max_values].copy()
        self._controller_obj = None
        self.view = widgets.VBox([])
        self.df = self._df.copy()
        self._build_view()

    def _build_view(self):
        self._build_filters(self.df)
        self._build_data_selector(self.df)

    def _build_data_selector(self, df):
        index = df.index
        style = {'description_width': '20px'}
        data_select = widgets.SelectMultiple(
            options=list(index.values),
            description='ID:',
            disabled=False,
            style=style
        )
        self._controller_obj = data_select
        self.view.children = list(self.view.children) + [data_select]

    @staticmethod
    def _get_selector_layout(values):
        if len(values) > 3:
            return widgets.Layout(height='80px')
        else:
            return widgets.Layout(height='{}px'.format(len(values) * 20))

    @staticmethod
    def _filter_df(df: pd.DataFrame,
                   field: str,
                   value: List,
                   allow_nan: bool = True):
        """Utility function the filters pandas.DataFrame to only
        include range of values for field when field is float,
        othewise filter pandas.DataFrame to only include
        select values for given field.
        Args:
            df (pandas.DataFrame) : dataframe to be filtered
            field (str): field in dataframe to be filtered
            value (List): values to filter dataframe by
        Kwargs:
            allow_nan (bool): allows nan values in filter
        """
        if is_float_dtype(df[field].dtype):
            min, max = value
            index_df = (df[field] >= min) & (df[field] <= max)
        else:
            index_df = df[field].isin(value)
        if allow_nan:
            index_df |= df[field].isna()
        return df.loc[index_df].copy()

    def _build_filters(self, df):
        def apply_button_cntrl(b):
            df = self._df
            for f in field_filters:
                if isinstance(f, widgets.FloatRangeSlider):
                    df = self._filter_df(df, f.description, f.value)
                else:
                    df = self._filter_df(df, f.description, f.value)
            self.df = df
            data_selector = self._controller_obj
            index = df.index[:self.max_values]
            list_index = list(index.values)
            result_length.value = "{} Result ID(s)".format(len(df))
            data_selector.options = list_index

        def reset_button_cntrl(b):
            for f in field_filters:
                if isinstance(f, widgets.FloatRangeSlider):
                    f.value = [f.min, f.max]
                else:
                    f.value = list(f.options)
            data_selector = self._controller_obj
            index = self._df.index[:self.max_values]
            list_index = list(index.values)
            result_length.value = "{} Result ID(s)".format(len(df))
            data_selector.options = list_index

        field_filters = []
        for field in df:
            values = df[field].unique()
            if is_float_dtype(df[field].dtype):
                values = values[~np.isnan(values)]
                field_filter = widgets.FloatRangeSlider(
                    value=[values.min(), values.max()],
                    min=values.min(),
                    max=values.max(),
                    step=0.1,
                    description=field,
                    disabled=False,
                    continuous_update=False,
                    orientation='horizontal',
                    readout=True,
                    readout_format='.1f',
                )
            else:
                style = {'description_width': '100px'}
                field_filter = widgets.SelectMultiple(
                    options=values,
                    value=values.tolist(),
                    description=field,
                    disabled=False,
                    style=style,
                    layout=self._get_selector_layout(values),
                )
            field_filters.append(field_filter)
        # Buttons to apply and reset filters
        apply_button = widgets.Button(description="Apply filters",
                                      layout=widgets.Layout(width='auto'))
        reset_button = widgets.Button(description="Reset filters",
                                      layout=widgets.Layout(width='auto'))
        apply_button.on_click(apply_button_cntrl)
        reset_button.on_click(reset_button_cntrl)
        buttons = [widgets.HBox([apply_button, reset_button])]
        divider = widgets.HTML('<hr>')
        result_length = widgets.HTML()
        self.view.children = (field_filters + buttons +
                              [divider] + [result_length])

    @property
    def controller_obj(self):
        return self._controller_obj


class ErrorFilterWidget(_FilterWidget):
    """Class that builds Error Analysis filter UI from pandas.DataFrame. Each
    field with float data type is given a slider to filter dataframe indices
    based on defined range. Other data types are given a select multiple
    dropdown to filter  dataframe indices based selected values.

    Args:
        df (pandas.DataFrame) : dataframe of filtering interface to be
            displayed
        max_values (int) : max number of values supported by interface

    Attributes:
        title (str) : title of interface to be displayed
        view (widgets.VBox) : widget class to display in Jupyter enviornment
        max_values (int) : max number of values supported by interface
    """
    def __init__(self, y_true, y_pred, index, max_values=2000):
        self.title = "Error Analysis"
        df = pd.DataFrame({"Actual Value": y_true, "Predicted Value": y_pred}, index=index)
        super().__init__(df, max_values=max_values)


class ShapFilterWidget(_FilterWidget):
    """Class that builds SHAP Analysis filter UI from pandas.DataFrame. Each
    field with float data type is given a slider to filter dataframe indices
    based on defined range. Other data types are given a select multiple
    dropdown to filter  dataframe indices based selected values.

    Args:
        df (pandas.DataFrame) : dataframe of filtering interface to be
            displayed
        max_values (int) : max number of values supported by interface

    Attributes:
        title (str) : title of interface to be displayed
        view (widgets.VBox) : widget class to display in Jupyter enviornment
        max_values (int) : max number of values supported by interface
    """
    def __init__(self, shap_values, columns, index, max_values=2000):
        self.title = "Impact Analysis"
        df = pd.DataFrame(shap_values, columns=columns, index=index)
        super().__init__(df, max_values=max_values)


class FeatureFilterWidget(_FilterWidget):
    """Class that builds Feature Value Analysis filter UI from
    pandas.DataFrame. Each field with float data type is given a slider to
    filter dataframe indices based on defined range. Other data types are
    given a select multiple dropdown to filter  dataframe indices based
    selected values.

    Args:
        df (pandas.DataFrame) : dataframe of filtering interface to be
            displayed
        max_values (int) : max number of values supported by interface

    Attributes:
        title (str) : title of interface to be displayed
        view (widgets.VBox) : widget class to display in Jupyter enviornment
        max_values (int) : max number of values supported by interface
    """
    def __init__(self, df, columns, max_values=2000):
        self.title = "Feature Analysis"
        df = df[columns]
        super().__init__(df, max_values=max_values)


class ShapUI:
    """Builds user interface with three tabs with various filters.
    Error Analysis allows the user to filter datapoints based on their
    error type. Impact Analysis allows users to filter datapoints based
    on their associated SHAP impact score for top important features.
    Feature Analysis allows users to filter datapoints for top important
    feature values. SHAP waterfall plot is displayed for each selected
    feature.

    Args:
        df (pandas.DataFrame) : dataframe containing feature values for dataset
        shap_values (np.array) : array containing shap values for dataset
        expected_value (float) : expected value or average shap value of
            explaination model
        y_true (Sequence) : sequence containing true values for dataset
        y_pred (Sequence) : sequence containing predicted values for dataset

    Kwargs:
        max_values (int) : max values supported by interface (defualt is 2000)
        max_features (int) : numer of top features to consider (defualt is 10)
    """
    def __init__(self,
                 df,
                 shap_values,
                 expected_value,
                 y_true,
                 y_pred,
                 max_values=2000,
                 max_features=10):

        self.max_values = max_values
        self.max_features = max_features
        self.expected_value = expected_value
        self.y_true = y_true
        self.y_pred = y_pred
        self.shap_values = shap_values
        self.df = df
        self.features = self.df.columns
        self._info_panel = shap_widget_info_panel
        self.controller_objs = []
        self.left_view = None
        self.right_view = None
        self.view = None

    def _build(self):
        self._get_top_features()
        self._transform_df()
        self._build_view()
        self._add_controllers(self.df)

    def _get_top_features(self):
        df = pd.DataFrame(self.shap_values,
                          columns=self.features,
                          index=self.df.index)
        top_features = (df
                        .abs()
                        .mean()
                        .argsort()
                        .argsort()
                        .sort_values(ascending=False)[:self.max_features])
        self.features = top_features.index
        self.shap_values = self.shap_values[:, top_features.values]
        self.df = self.df[self.features]

    def _build_view(self):
        # Filter widget Views
        uis = [FeatureFilterWidget(self.df,
                                   self.features,
                                   max_values=self.max_values),
               ErrorFilterWidget(self.y_true,
                                 self.y_pred,
                                 self.df.index,
                                 max_values=self.max_values),
               ShapFilterWidget(self.shap_values,
                                self.features,
                                self.df.index,
                                max_values=self.max_values)]

        self.controller_objs = [ui.controller_obj for ui in uis]
        self.left_view = widgets.Tab([ui.view for ui in uis],
                                     layout=widgets.Layout(width='40%'))
        for i, ui in enumerate(uis):
            self.left_view.set_title(i, ui.title)
        self.right_view = widgets.Tab(layout=widgets.Layout(width='60%'))
        self.view = widgets.HBox([self.left_view, self.right_view])

    def _transform_df(self):
        # Filter to max allowed values
        self.df = self.df[:self.max_values].copy()
        self.shap_values = self.shap_values[:self.max_values]
        self.y_true = self.y_true[:self.max_values]
        self.y_pred = self.y_pred[:self.max_values]
        # Add Shap values and groud truth for ease of indexing
        self.df['_shap_values'] = list(self.shap_values)
        self.df['y_true'] = self.y_true

    def show(self):
        """Displays SHAP UI in Jupyter enviornment.
        """
        self._build()
        display(self.view)

    def info(self):
        info_panel = InfoPanel(**self._info_panel)
        info_panel.show()

    def _build_figs(self, df):
        figs = []
        for pid, row in df.iterrows():
            fig = shap_waterwall_plot(self.expected_value,
                                      row._shap_values,
                                      row.values,
                                      row.index,
                                      y_true=row.y_true)
            fig.update_layout(title="SHAP Values for {}".format(pid))
            figs.append(go.FigureWidget(fig))
        return figs

    def _add_controllers(self, df):
        def select_plot_controller(change):
            pids = list(change.owner.value)
            self.right_view.children = self._build_figs(df.loc[pids])
            for i, pid in enumerate(pids):
                self.right_view.set_title(i, str(pid))
        for obj in self.controller_objs:
            obj.observe(select_plot_controller, names=['value'], type='change')
