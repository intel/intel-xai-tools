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
from persist import persist, load_widget_state
from base64 import b64encode
from utils import *

st.set_page_config(initial_sidebar_state="expanded")
if "overview_graphics" not in st.session_state:
    st.session_state["overview_graphics"] = []

if "overview_graphics_dict" not in st.session_state:
    st.session_state["overview_graphics_dict"] = {}


def main():
    navigate_menu()
    st.markdown("# Model Details")
    st.write("This section provides a general, high-level description or the metadata of the model.")
    warning_placeholder = st.empty()
    st.text_input(
        r"$\textsf{\LARGE  Model Name}$",
        value=st.session_state.get("model_name"),
        help="The name of the model.",
        key=persist("model_name"),
    )
    model_name = st.session_state.model_name or None
    do_warn = False
    warning_msg = "Warning: The following field is required but is not filled in: "
    if not model_name:
        warning_msg += " Model Name"
        do_warn = True
    if do_warn:
        warning_placeholder.error(warning_msg)

    st.write("\n")
    st.text_input(
        r"$\textsf{\LARGE  Model Path}$",
        value=st.session_state.get("path"),
        help="Provide the path where the model is stored.",
        key=persist("path"),
    )
    st.write("\n")
    st.text_area(
        r"$\textsf{\LARGE  Model Card Overview}$",
        value=st.session_state.get("overview"),
        placeholder="Short description of the model card.",
        help="A brief, description of the model card.",
        key=persist("overview"),
    )
    st.write("\n")
    st.write(r"$\textsf{\LARGE  Model Version}$")
    with st.container(border=True):

        st.text_input(
            r"$\textsf{\large  Version Name}$",
            value=st.session_state.get("model_version_name"),
            help="The version of the model.",
            key=persist("model_version_name"),
        )
        st.text_input(
            r"$\textsf{\large  Difference from the previous version}$",
            value=st.session_state.get("model_version_diff"),
            help="The changes from the previous model version.",
            key=persist("model_version_diff"),
        )
        st.text_input(
            r"$\textsf{\large  Date}$",
            value=st.session_state.get("model_version_date"),
            help="The date when the model version was released.",
            key=persist("model_version_date"),
        )

    combine_version_info()
    st.write("\n")
    st.write(r"$\textsf{\LARGE Licenses}$")

    generate_dynamic_form(
        number_label="Licenses",
        selectbox_range=(0, 11),
        selectbox_index_key="licenses_count",
        selectbox_help="Number of Licenses.",
        item_label="License",
        first_input_label="Identifier",
        second_input_label="Custom License Text",
        first_input_key="license_id",
        second_input_key="license_text",
        first_input_help="A standard SPDX license identifier (https://spdx.org/licenses/),"
        "or proprietary for an unlicensed module.",
        second_input_help="Text of a custom license.",
        form_key="license_form",
        submit_button_label="Submit Licenses",
        first_field_key="identifier",
        second_field_key="custom_text",
        session_key="licenses",
        processing_function=combine_info,
    )

    st.write("\n")
    st.text_area(
        r"$\textsf{\LARGE Model Documentation}$",
        value=st.session_state.get("documentation"),
        placeholder="A thorough description of the model and its usage.",
        help="Model Documentation includes model's general information, its usage, version, details "
        "regarding its implementation whether it is borrowed or an original architecture, version. "
        "Any disclaimer or copyright should also be mentioned here.",
        key=persist("documentation"),
    )

    st.write("\n")
    st.write(r"$\textsf{\LARGE  Owners}$")

    generate_dynamic_form(
        number_label=" Model owners",
        selectbox_range=(0, 11),
        selectbox_index_key="owners_count",
        selectbox_help="Number of individuals or teams who own the model.",
        item_label="Owner",
        first_input_label="Name of owner",
        second_input_label="Contact of owner",
        first_input_key="name",
        second_input_key="contact",
        first_input_help="Name of the model owner",
        second_input_help="The contact information for the model owner or owners. "
        "These could be individual email addresses, a team mailing list expressly, or a monitored feedback form.",
        form_key="owners_form",
        submit_button_label="Submit Model owners",
        first_field_key="name",
        second_field_key="contact",
        session_key="owners",
        processing_function=combine_info,
    )
    st.write("\n")
    st.write(r"$\textsf{\LARGE  Citations}$")
    generate_dynamic_form(
        number_label="Citations",
        selectbox_range=(0, 21),
        selectbox_index_key="citations_count",
        selectbox_help="Number of citations to reference this model.",
        item_label="Citation",
        first_input_label="Style",
        second_input_label="Citation",
        first_input_key="cite_style",
        second_input_key="cite",
        first_input_help="The citation style, such as MLA, APA, Chicago, or IEEE.",
        second_input_help="Citation to refer to the model.",
        form_key="citations_form",
        submit_button_label="Submit Citations",
        first_field_key="style",
        second_field_key="citation",
        session_key="citations",
        processing_function=combine_info,
    )

    st.write("\n")
    st.write(r"$\textsf{\LARGE  References}$")
    generate_dynamic_form(
        number_label="References",
        selectbox_range=(0, 16),
        selectbox_index_key="references_count",
        selectbox_help="Number of links to provide more information about the model.",
        item_label="Reference",
        first_input_label="Reference",
        second_input_label="",
        first_input_key="reference",
        second_input_key="",
        first_input_help="Links providing more information about the model. "
        "You can link to foundational research, technical documentation, or other materials that may be"
        "useful to your audience.",
        second_input_help="",
        form_key="references_form",
        submit_button_label="Submit References",
        first_field_key="reference",
        second_field_key="",
        session_key="references",
        processing_function=parse_references_to_list,
    )
    st.write("\n")

    st.write(r"$\textsf{\LARGE  Overview Graphics}$")
    st.text_area(
        r"$\textsf{\Large  Graphics Description}$",
        value=st.session_state.get("overview_graphic_desp"),
        placeholder="A description of this collection of overview graphics.",
        help="A description of this collection of overview graphics.",
        key=persist("overview_graphic_desp"),
    )

    uploaded_overview_graphics = {}
    uploaded_files = st.file_uploader(
        r"$\textsf{\Large Upload Overview Graphics}$",
        type=["png"],
        help="Please choose an image(s) (.png) to upload",
        accept_multiple_files=True,
    )

    for uploaded_file in uploaded_files:
        # Read the contents of the uploaded file
        data_uri = b64encode(uploaded_file.getvalue()).decode("utf-8")

        # Create the HTML tag with the base64 encoded data
        img_tag = f"{data_uri}"
        uploaded_overview_graphics[uploaded_file.name[:-4]] = img_tag

    st.session_state["overview_graphics_dict"].update(uploaded_overview_graphics)
    update_graphics_list_from_dict("overview_graphics_dict", "overview_graphics")

    if st.button("Clear Uploaded Files", help="Clear all the uploaded overview graphics"):
        st.session_state["overview_graphics"] = []


if __name__ == "__main__":
    load_widget_state()
    main()