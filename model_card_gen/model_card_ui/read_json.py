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

"""This module reads an uploaded json file to directory to be used by jinja parser function.
Fields are in form of JSON_field: modelcardUI_field"""

import streamlit as st
import json
from jsonschema import ValidationError
from utils import (
    convert_str_to_key_value_map,
    initialize_consideration_data,
    convert_key_value_map_to_str,
    initialize_session_with_nested_data,
)
from intel_ai_safety.model_card_gen.validation import validate_json_schema


def read_uploaded_file(uploaded_file):
    uploaded_json_string = uploaded_file.getvalue().decode("utf-8")
    json_data = json.loads(uploaded_json_string)
    warning_placeholder = st.empty()
    try:
        validate_json_schema(json_data)
    except (ValidationError, ValueError) as e:
        warning_placeholder.error(
            "Warning: The schema version of the uploaded JSON does not correspond to a model card schema version or "
            "the uploaded JSON does not follow the model card schema."
        )

    st.session_state.model_card_json = json_data
    st.session_state["schema_version"] = json_data.get("schema_version", "")

    model_details_json = json_data.get("model_details", {})
    st.session_state["model_name"] = model_details_json.get("name", "")
    st.session_state["path"] = model_details_json.get("path", "")
    st.session_state["overview"] = model_details_json.get("overview", "")
    st.session_state["documentation"] = model_details_json.get("documentation", "")
    st.session_state["model_version_name"] = model_details_json.get("version", {}).get("name", "")
    st.session_state["model_version_date"] = model_details_json.get("version", {}).get("date", "")
    st.session_state["model_version_diff"] = model_details_json.get("version", {}).get("diff", "")
    st.session_state["overview_graphic_desp"] = model_details_json.get("graphics", {}).get("description", "")
    st.session_state["overview_graphics"] = model_details_json.get("graphics", {}).get("collection", [])

    license_info_keys = {"identifier": "license_id", "custom_text": "license_text"}
    initialize_session_with_nested_data(model_details_json, "licenses", "licenses", license_info_keys)

    owner_info_keys = {"name": "name", "contact": "contact"}
    initialize_session_with_nested_data(model_details_json, "owners", "owners", owner_info_keys)

    reference_info_keys = {"reference": "reference"}
    initialize_session_with_nested_data(model_details_json, "references", "references", reference_info_keys)

    cite_info_keys = {"citation": "cite", "style": "cite_style"}
    initialize_session_with_nested_data(model_details_json, "citations", "citations", cite_info_keys)

    model_parameters_json = json_data.get("model_parameters", {})
    st.session_state["model_architecture"] = model_parameters_json.get("model_architecture", "")
    st.session_state["input_format"] = model_parameters_json.get("input_format", "")
    st.session_state["output_format"] = model_parameters_json.get("output_format", "")

    st.session_state["input_format_map"] = convert_str_to_key_value_map(
        convert_key_value_map_to_str(model_parameters_json.get("input_format_map", []), "input_format_map")
    )
    st.session_state["output_format_map"] = convert_str_to_key_value_map(
        convert_key_value_map_to_str(model_parameters_json.get("output_format_map", []), "output_format_map")
    )

    considerations_json = json_data.get("considerations", {})
    initialize_consideration_data(considerations_json.get("users", []), "users")
    initialize_consideration_data(considerations_json.get("use_cases", []), "use_cases")
    initialize_consideration_data(considerations_json.get("limitations", []), "limitations")
    initialize_consideration_data(considerations_json.get("tradeoffs", []), "tradeoffs")

    risk_info_keys = {"name": "risk_name", "mitigation_strategy": "strategy"}
    initialize_session_with_nested_data(
        considerations_json, "ethical_considerations", "ethical_considerations", risk_info_keys
    )

    quantitative_analysis_graphics_json = json_data.get("quantitative_analysis", {}).get("graphics", {})
    st.session_state["performance_graphic_desp"] = quantitative_analysis_graphics_json.get("graphics", {}).get(
        "description", ""
    )
    st.session_state["performance_graphics"] = quantitative_analysis_graphics_json.get("graphics", {}).get(
        "collection", []
    )

    st.success(f"Read uploaded Model Card File: {uploaded_file.name}")
    return json_data
