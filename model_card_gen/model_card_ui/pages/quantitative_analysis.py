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
from base64 import b64encode
from utils import save_uploaded_file_to_session_state, update_graphics_list_from_dict, navigate_menu

st.set_page_config(initial_sidebar_state="expanded")

if "performance_graphics" not in st.session_state:
    st.session_state["performance_graphics"] = []

if "performance_graphics_dict" not in st.session_state:
    st.session_state["performance_graphics_dict"] = {}


def main():
    navigate_menu()
    st.markdown("# Performance Quantitative Analysis")
    st.write("This section provides details regarding the model performance metrics being reported.")
    st.markdown(
        "If you are interested in viewing examples of metric files, please [click here](https://github.com/intel/intel-xai-tools/tree/bb7b1fa008580606eac19a58020e5937a275dbf7/model_card_gen/intel_ai_safety/model_card_gen/docs/examples/csv). "
        "To learn how to create these files or see a step-by-step example, you can follow this [link](https://github.com/intel/intel-xai-tools/blob/bb7b1fa008580606eac19a58020e5937a275dbf7/notebooks/model_card_gen/hugging_face_model_card/hugging-face-model-card.ipynb) "
        "for further guidance."
    )
    # Upload metrics by threshold
    metric_by_thresh_file = st.file_uploader(
        r"$\textsf{\Large Metrics By Threshold}$",
        type=["csv"],
        help="Please choose a CSV file to upload",
        accept_multiple_files=False,
    )
    if metric_by_thresh_file is not None:
        save_uploaded_file_to_session_state(metric_by_thresh_file, "metrics_by_threshold")

    st.write("\n")
    # Upload metrics by group
    metric_by_grp_file = st.file_uploader(
        r"$\textsf{\Large Metrics By Group}$",
        type=["csv"],
        help="Please choose a CSV file to upload",
        accept_multiple_files=False,
    )
    if metric_by_grp_file is not None:
        save_uploaded_file_to_session_state(metric_by_grp_file, "metrics_by_grp")
    st.write("\n")
    st.write(r"$\textsf{\LARGE Performance Graphics}$")
    st.text_area(
        r"$\textsf{\Large Description of Performance Graphics}$",
        value=st.session_state.get("performance_graphic_desp"),
        placeholder="A description of the collection of performance graphics.",
        help="A description of the collection of performance graphics.",
        key=persist("performance_graphic_desp"),
    )
    st.write("\n")
    uploaded_performance_graphics = {}
    uploaded_perf_files = st.file_uploader(
        r"$\textsf{\Large Upload Performance Graphics}$",
        type=["png"],
        help="Please upload any image(s) (.png) illustrating the model performance.",
        accept_multiple_files=True,
    )

    for uploaded_file in uploaded_perf_files:
        # Read the contents of the uploaded image and store it after encoding it in base64
        data_uri = b64encode(uploaded_file.getvalue()).decode("utf-8")
        uploaded_performance_graphics[uploaded_file.name[:-4]] = data_uri

    st.session_state["performance_graphics_dict"].update(uploaded_performance_graphics)
    update_graphics_list_from_dict("performance_graphics_dict", "performance_graphics")
    if st.button("Clear Uploaded Files", help="Clear all the uploaded performance graphics"):
        st.session_state["performance_graphics"] = []
        st.session_state["performance_graphics_dict"] = {}


if __name__ == "__main__":
    load_widget_state()
    main()