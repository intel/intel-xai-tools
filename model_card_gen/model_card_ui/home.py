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
from read_json import read_uploaded_file
from session_state_initializer import initialize_session_state
import plotly.io as pio
from utils import navigate_menu

st.set_page_config(initial_sidebar_state="expanded")
pio.templates.default = "plotly"


def upload_model_card():
    st.markdown("## Upload Model Card")
    st.write(
        "If you are interested in viewing an example, please [click here](https://github.com/intel/intel-xai-tools/tree/main/model_card_gen/intel_ai_safety/model_card_gen/docs/examples/json)."
    )
    uploaded_file = st.file_uploader("", type=["json"], help="Please choose a JSON (.json) file type to upload")
    if uploaded_file is not None:
        read_uploaded_file(uploaded_file)


def main_page():
    st.header("Model Card Generator")
    st.markdown(
        "This tool allows users to create interactive HTML reports containing model performance and fairness metrics "
        "using a simple interface."
    )
    st.markdown(
        "To begin, you can either upload an existing Model Card in JSON format below or input your model card details by selecting the respective sections from the sidebar."
    )
    upload_model_card()


if __name__ == "__main__":
    load_widget_state()
    initialize_session_state()
    navigate_menu()
    if "runpage" not in st.session_state:
        st.session_state.runpage = main_page
    st.session_state.runpage()