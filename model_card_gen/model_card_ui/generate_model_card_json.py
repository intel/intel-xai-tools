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
from utils import remove_empty_values, form_desp_list, convert_str_to_key_value_map
from intel_ai_safety.model_card_gen.validation import validate_json_schema

def generate_json_template():
    """
    Build the Model Card JSON using the session state values for different keys.
    """

    mc = {
        "schema_version": st.session_state.get("schema_version"),
        "model_details": {
            "name": st.session_state.get("model_name"),
            "overview": st.session_state.get("overview"),
            "documentation": st.session_state.get("documentation"),
            "owners": st.session_state.get("owners"),
            "version": {
                "name": st.session_state.get("model_version_name"),
                "diff": st.session_state.get("model_version_diff"),
                "date": st.session_state.get("model_version_date"),
            },
            "licenses": st.session_state.get("licenses"),
            "references": st.session_state.get("references"),
            "citations": st.session_state.get("citations"),
            "documentation": st.session_state.get("documentation"),
            "path": st.session_state.get("path"),
            "graphics": {
                "description": st.session_state.get("overview_graphic_desp"),
                "collection": st.session_state.get("overview_graphics"),
            },
        },
        "model_parameters": {
            "model_architecture": st.session_state.get("model_architecture"),
            "input_format": st.session_state.get("input_format"),
            "input_format_map": convert_str_to_key_value_map(st.session_state.get("input_format_map_text")),
            "output_format": st.session_state.get("output_format"),
            "output_format_map": convert_str_to_key_value_map(st.session_state.get("output_format_map_text"))#st.session_state.get("output_format_map"),
        },
        "quantitative_analysis": {
            "graphics": {
                "description": st.session_state.get("performance_graphic_desp"),
                "collection": st.session_state.get("performance_graphics"),
            }
        },
        "considerations": {
            "users": form_desp_list(st.session_state.get("users_text"), session_key = "users"),
            "use_cases": form_desp_list(st.session_state.get("use_cases_text"), session_key = "use_cases"),
            "limitations": form_desp_list(st.session_state.get("limitations_text"), session_key = "limitations"),
            "tradeoffs": form_desp_list(st.session_state.get("tradeoffs_text"), session_key = "tradeoffs"),
            "ethical_considerations": st.session_state.get("ethical_considerations"),
        },
    }
    mc_cleaned = remove_empty_values(mc)
    validate_json_schema(mc_cleaned)
    st.session_state["model_card_json"] = mc_cleaned
    return mc_cleaned


def main():
    return