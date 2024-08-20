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
from utils import combine_info, generate_dynamic_form, navigate_menu

st.set_page_config(initial_sidebar_state="expanded")


def main():
    navigate_menu()

    st.markdown("# Considerations")
    st.write(
        "Describe the considerations that should be taken into account regarding the model's construction, training, "
        "and application."
    )
    users = st.text_input(
        r"$\textsf{\LARGE  Users}$",
        value=st.session_state.get("users_text"),
        help="Who are the intended users of the model? This may include researchers, developers, and/or clients. "
        "You might also include information about the downstream users you expect to interact with your model.",
        key=persist("users_text"),
    )
    st.write("\n")
    use_cases = st.text_area(
        r"$\textsf{\LARGE  Use Cases}$",
        value=st.session_state.get("use_cases_text"),
        placeholder="Intended use cases of the model.",
        help="What are the intended use cases of the model? What use cases are out-of-scope?",
        key=persist("use_cases_text"),
    )
    st.write("\n")
    limitations = st.text_area(
        r"$\textsf{\LARGE  Limitations}$",
        value=st.session_state.get("limitations_text"),
        placeholder="Technical Limitations of the model.",
        help="What are the known limitations of the model? This may include technical limitations,"
        "or conditions that may degrade model performance.",
        key=persist("limitations_text"),
    )
    st.write("\n")
    tradeoffs = st.text_area(
        r"$\textsf{\LARGE  Tradeoffs}$",
        value=st.session_state.get("tradeoffs_text"),
        placeholder="Tradeoffs of the model.",
        help="What are the known accuracy/performance tradeoffs for the model?",
        key=persist("tradeoffs_text"),
    )
    st.write("\n")
    st.write(r"$\textsf{\LARGE  Ethical Considerations}$")
    st.write(
        "What are the ethical risks involved in application of this model? "
        "For each risk, you may also provide a mitigation strategy that you've implemented, or "
        "one that you suggest to users."
    )

    generate_dynamic_form(
        number_label="Risks",
        selectbox_range=(0, 11),
        selectbox_index_key="ethical_considerations_count",
        selectbox_help="Number of risks",
        item_label="Risk",
        first_input_label="Name of Risk",
        second_input_label="Mitigation Strategy",
        first_input_key="risk_name",
        second_input_key="strategy",
        first_input_help="What are the ethical risks involved in application of this model?",
        second_input_help="For each risk, you may also provide a mitigation strategy that you've implemented, or "
        "one that you suggest to users.",
        form_key="risks_form",
        submit_button_label="Submit Risks",
        first_field_key="name",
        second_field_key="mitigation_strategy",
        session_key="ethical_considerations",
        processing_function=combine_info,
    )


if __name__ == "__main__":
    load_widget_state()
    main()