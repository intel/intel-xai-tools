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


def initialize_session_state():
    """
    Initializes the Streamlit session state with default values for model card information and related details.
    """
    if "model_name" not in st.session_state:

        st.session_state.update(
            {
                # Model card information
                "model_card_json": {},  # JSON representation of the model card
                "model_card_html": "",  # HTML representation of the model card
                # Model Details
                "model_name": "",  # Name of the model
                "path": "",  # Path to the model or related resources
                "overview": "",  # Overview description of the model
                "schema_version": "",  # Version of the model card schema
                "model_version_name": "",  # Name of the specific model version
                "model_version_diff": "",  # Description of differences from previous versions
                "model_version_date": "",  # Release date of the model version
                "licenses": [],  # List of licenses associated with the model
                "licenses_count": 0,  # Initial count of licenses
                "license_info": {},  # Dictionary containing license information
                "documentation": "",  # Documentation associated with the model
                "owners": [],  # List of owners or maintainers of the model
                "owners_count": 0,  # Initial count of owners
                "owner_info": {},  # Dictionary containing owner information
                "citations": [],  # List of citations for the model
                "citations_count": 0,  # Initial count of citations
                "cite_info": {},  # Dictionary containing citation information
                "references": [],  # List of references related to the model
                "references_count": 0,  # Textual description of references
                "overview_graphic_desp": "",  # Description of overview graphics
                "overview_graphics_dict": {},  # Dictionary of overview graphics
                "overview_graphics": [],  # List of overview graphics
                # Model Parameters
                "model_architecture": "",  # Description of the model's architecture
                "input_format": "",  # Expected input format for the model
                "input_format_map": [],  # Mapping of input formats
                "input_format_map_text": "",  # Textual description of input format mapping
                "output_format": "",  # Expected output format from the model
                "output_format_map": [],  # Mapping of output formats
                "output_format_map_text": "",  # Textual description of output format mapping
                # Considerations
                "users": [],  # List of intended users of the model
                "users_text": "",  # Textual description of intended users
                "use_cases": [],  # List of intended use cases for the model
                "use_cases_text": "",  # Textual description of use cases
                "limitations": [],  # List of known limitations of the model
                "limitations_text": "",  # Textual description of limitations
                "tradeoffs": [],  # List of trade-offs considered in the model design
                "tradeoffs_text": "",  # Textual description of trade-offs
                "ethical_considerations_count": 0,  # Count of ethical considerations
                "ethical_considerations": [],  # List of strategies to mitigate risks
                # Data Analysis
                "metrics_by_grp": "",  # Metrics broken down by group
                "metrics_by_threshold": "",  # Metrics at different thresholds
                "performance_graphic_desp": "",  # Description of performance graphics
                "performance_graphics_dict": {},  # Dictionary of performance graphics
                "performance_graphics": [],  # Performance graphics image collection
            }
        )
