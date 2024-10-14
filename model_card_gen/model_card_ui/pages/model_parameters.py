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

import streamlit as st
from persist import persist, load_widget_state
from utils import convert_str_to_key_value_map, navigate_menu

st.set_page_config(initial_sidebar_state="expanded")


def main():
    navigate_menu()
    st.markdown("# Model Parameters")
    st.write("This section provides details regarding the parameters for construction of the model.")
    st.text_area(
        r"$\textsf{\LARGE  Model Architecture}$",
        value=st.session_state.get("model_architecture"),
        placeholder="The architecture of the model.",
        help="The architecture of the model.",
        key=persist("model_architecture"),
    )
    st.write("\n")
    st.text_area(
        r"$\textsf{\LARGE  Input Format}$",
        value=st.session_state.get("input_format"),
        help="The data format for inputs to the model.",
        key=persist("input_format"),
    )
    st.write("\n")
    st.text_area(
        r"$\textsf{\LARGE  Input Format Map}$",
        value=st.session_state.get("input_format_map_text"),
        placeholder="Input data format in the form of list (comma seperated) of key-value pairs. For example, image_data: RGB images ('PNG'), text: String data",
        help="The data format for inputs to the model, in key-value format.",
        key=persist("input_format_map_text"),
    )
    st.write("\n")
    st.text_area(
        r"$\textsf{\LARGE  Output Format}$",
        st.session_state.get("output_format"),
        help="The data format for outputs from the model.",
        key=persist("output_format"),
    )
    st.write("\n")
    st.text_area(
        r"$\textsf{\LARGE  Output Format Map}$",
        value=st.session_state.get("output_format_map_text"),
        placeholder="Output data format in the form of list (comma seperated) of key-value pairs. For example, classification_probabilities: Probability (between 0 to 1) of the example belonging to each class.",
        help="The data format for outputs from the model, in key-value format.",
        key=persist("output_format_map_text"),
    )


if __name__ == "__main__":
    load_widget_state()
    main()