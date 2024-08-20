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
"""
This module builds the Home Page of the Model Card Generator interactive UI allowing users to
create, upload, and download Model Cards.
"""
import streamlit as st
from persist import load_widget_state
import json
from read_json import read_uploaded_file
from generate_model_card_json import generate_json_template
from model_card_html_generator import generate_model_card_html
from session_state_initializer import initialize_session_state
import streamlit.components.v1 as components
import plotly.io as pio
from utils import navigate_menu

st.set_page_config(initial_sidebar_state="expanded")
pio.templates.default = "plotly"


def view_model_card():
    navigate_menu()
    model_card = generate_json_template()

    st.markdown("# Current Model Card")

    if st.session_state.get("model_name") is None or st.session_state.model_name == "":
        st.error("Please fill model details or upload a Model Card to view the Model Card.")
    else:
        if st.session_state.get("metrics_by_threshold"):
            metric_threshold_csv = st.session_state.get("metrics_by_threshold")
        else:
            metric_threshold_csv = None

        if st.session_state.get("metrics_by_grp"):
            metric_grp_csv = st.session_state.get("metrics_by_grp")
        else:
            metric_grp_csv = None
        model_card_html = generate_model_card_html(model_card, metric_threshold_csv, metric_grp_csv)
        html_content = model_card_html.replace("<head>", "<head><style>body { background-color: white; }</style>")
        if st.session_state.get("model_card_json") != {}:
            left, mid_left, mid_right, right = st.columns([2, 5, 5, 2])
            model_card_json_str = json.dumps(st.session_state.get("model_card_json"), indent=2)
            with mid_left:
                mc_html_download_button = st.download_button(
                    label="Download Model Card (HTML)",
                    data=html_content,
                    file_name="Model_Card.html",
                    help="The current Model Card will be downloaded as a HTML (.html) file",
                )

            with mid_right:
                mc_json_download_button = st.download_button(
                    label="Download Model Card (JSON)",
                    data=model_card_json_str,
                    file_name="Model_Card.json",
                    help="The current Model Card will be downloaded as a JSON (.json) file",
                )
            if mc_html_download_button:
                st.success("Your current Model Card (HTML) has been successfully downloaded!🎉")
            if mc_json_download_button:
                st.success("Your current Model Card (JSON) has been successfully downloaded!🎉")

        components.html(html_content, scrolling=True, height=2000, width=750)


if __name__ == "__main__":
    load_widget_state()
    view_model_card()