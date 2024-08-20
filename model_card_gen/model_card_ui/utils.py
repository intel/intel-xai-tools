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
from persist import persist


def save_uploaded_file_to_session_state(uploaded_file, session_state_field):
    """
    Saves an uploaded file to the server's filesystem and updates the session state with the file path.

    Parameters:
    uploaded_file (UploadedFile): The file uploaded by the user through Streamlit's file_uploader.
    session_state_field (str): The session state key whose file path will be stored.

    Returns:
    None
    """
    file_path = uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.read())
    st.session_state[session_state_field] = file_path
    st.success(f"Saved File: {uploaded_file.name}")


def update_graphics_list_from_dict(graphics_dict, graphics_list):
    """
    Updates a list of graphics in the session state by adding new entries from the provided dictionary of graphics.

    Parameters:
    graphics_dict (str): The session state key that references the dictionary of graphics.
                          Each entry in the dictionary is in the format {image_file_name: image_base64_str}.

    graphics_list (str): The session state key that references the list of graphics.
                         Each entry in the list is a dictionary in the format {"name": image_file_name, "image": image_base64_str}.

    Returns:
    None
    """
    if st.session_state.get(graphics_dict) is not None and st.session_state.get(graphics_dict) != {}:
        for file_name, img_str in st.session_state.get(graphics_dict).items():
            st.session_state[graphics_list].append({"name": file_name, "image": img_str})


def combine_version_info():
    """
    Combines individual model version details from the Streamlit session state into a single dictionary
    under the 'model_version' key in the session state.
    """
    st.session_state.model_version = {
        "name": st.session_state.get("model_version_name"),
        "diff": st.session_state.get("model_version_diff"),
        "date": st.session_state.get("model_version_date"),
    }


def parse_references_to_list(reference_key, reference_texts, session_key):
    """
    Parses a list of reference texts into a list of dictionaries to store it in the Streamlit session state.

    Parameters:
    reference_key (str): The key to be used for each reference in the resulting dictionaries.
    reference_texts (list of str): The list of reference texts to be parsed.
    session_key (str): The key under which the resulting list of dictionaries will be stored in the session state.
    """
    references_list = [{reference_key: reference_text} for reference_text in reference_texts if reference_text]
    st.session_state[session_key] = references_list


def combine_info(first_field_key, first_field_values, second_field_key, second_field_values, session_key):
    """
    Combines two lists of field values into a list of dictionaries and stores it in the Streamlit session state.

    Parameters:
    first_field_key (str): The key associated with the first list of field values.
    first_field_values (list): The list of values for the first field.
    second_field_key (str): The key associated with the second list of field values.
    second_field_values (list): The list of values for the second field.
    session_key (str): The key under which the combined list of dictionaries will be stored in the session state.

    If the session_key is 'licenses', the function creates a list of dictionaries with only the second field key
    if the second field value is present; otherwise, it uses the first field key. For other session keys, it
    combines both field values into dictionaries with both keys.
    """
    if session_key == "licenses":
        combined_info = [
            {second_field_key: second_field_value} if second_field_value else {first_field_key: first_field_value}
            for first_field_value, second_field_value in zip(first_field_values, second_field_values)
        ]
    else:
        combined_info = [
            {first_field_key: first_field_value, second_field_key: second_field_value}
            for first_field_value, second_field_value in zip(first_field_values, second_field_values)
            if first_field_value != "" or second_field_value != ""
        ]

    st.session_state[session_key] = combined_info


def form_desp_list(input_desp,session_key):
    """
    Splits a comma-separated string of descriptions into a list of dictionaries with a 'description' key.

    Parameters:
    input_desp (str): A comma-separated string of descriptions.
    session_key (str): The key used to update the corresponding value in the session state with the list of descriptions. 

    Returns:
    list of dict: A list of dictionaries, each containing a single 'description' key-value pair.
    """
    if input_desp is None or input_desp == "":
        return []
    if session_key == "users":
        desp_list = input_desp.split(",")
    else:
        desp_list = input_desp.split("\n")
    desp_list = [{"description": desp.strip()} for desp in desp_list if desp.strip() != ""]
    return desp_list


def generate_dynamic_form(
    number_label,
    selectbox_range,
    selectbox_index_key,
    selectbox_help,
    item_label,
    first_input_label,
    second_input_label,
    first_input_key,
    second_input_key,
    first_input_help,
    second_input_help,
    form_key,
    submit_button_label,
    first_field_key,
    second_field_key,
    session_key,
    processing_function,
):
    left, right = st.columns([6, 3])
    with st.container():
        with left:
            st.write("\n")
            st.write("\n")
            st.markdown(rf"$\textsf{{\large Number of {number_label}}}$")
        with right:
            if selectbox_range[0] == 0:
                number_of_items = st.selectbox(
                    "",
                    list(range(*selectbox_range)),
                    index=st.session_state.get(selectbox_index_key, selectbox_range[0]),
                    help=selectbox_help,
                    key=persist(selectbox_index_key),
                )
            else:
                number_of_items = st.selectbox(
                    "",
                    list(range(*selectbox_range)),
                    index=st.session_state.get(selectbox_index_key, selectbox_range[0] - 1),
                    help=selectbox_help,
                    key=persist(selectbox_index_key),
                )
    if number_of_items > 0:
        with st.form(key=form_key):
            first_field_values = []
            if item_label != "Reference":

                second_field_values = []
            for i in range(number_of_items):
                st.markdown(f"### {item_label} {i+1}")

                current_first = st.session_state.get(f"{first_input_key.lower()}_{i}", "")

                first = st.text_input(
                    f"{first_input_label}",
                    value=current_first,
                    help=first_input_help,
                    key=persist(f"{first_input_key.lower()}_{i}"),
                )
                if item_label != "Reference":
                    current_second = st.session_state.get(f"{second_input_key.lower()}_{i}", "")
                    second = st.text_input(
                        f"{second_input_label}",
                        value=current_second,
                        help=second_input_help,
                        key=persist(f"{second_input_key.lower()}_{i}"),
                    )

                if item_label == "License":
                    if first and second:
                        st.warning(
                            f"Please fill out only one of the fields: either '{first_input_label}' or '{second_input_label}'."
                        )
                        valid_input = False
                    elif first or second:
                        valid_input = True
                    else:
                        st.info(f"Please fill out one of the fields above.")
                        valid_input = False

                    if valid_input:
                        first_field_values.append(first)
                        second_field_values.append(second)
                elif item_label == "Reference":
                    first_field_values.append(first)
                else:
                    first_field_values.append(first)
                    second_field_values.append(second)
            if st.form_submit_button(label=submit_button_label):
                if item_label == "Reference":
                    processing_function(first_field_key, first_field_values, session_key)
                else:
                    if not all(not fi for fi in first_field_values) or not all(not sec for sec in second_field_values):
                        processing_function(
                            first_field_key, first_field_values, second_field_key, second_field_values, session_key
                        )

def convert_key_value_map_to_str(json_data, json_key):
    """
    Converts a list of dictionaries to a key-value string and updates the Streamlit session state.

    Parameters:
    json_data (dict): The nested json data containing lists of key-value mapping.
    list_key (str): The key for accessing the list of key-value mapping within the parent data.

    The function concatenates all key-value pairs from the dictionaries into a single string,
    separated by commas, and stores it in the session state under a new key with "_text" appended.
    """
    key_values = [d["key"] + " : " + d["value"] for d in json_data]
    key_value_string = ", ".join(key_values)
    st.session_state[json_key + "_text"] = key_value_string
    return key_value_string

def convert_str_to_key_value_map(key_value_string):
    """
    Converts a string representation of key-value pairs into a list of dictionaries.

    Parameters:
    key_value_string (str): A string containing key-value pairs separated by commas,
                            with each key and value separated by a colon.

    Returns:
    list of dict: A list of dictionaries, each representing a key-value pair from the input string.
    """
    if not key_value_string:
        return []

    key_value_pairs = key_value_string.split(",")
    key_value_list = []

    for pair in key_value_pairs:
        trimmed_pair = pair.strip()
        if not trimmed_pair:
            continue

        key_and_value = trimmed_pair.split(":")
        if len(key_and_value) == 2 and key_and_value[0] and key_and_value[1]:
            key_value_dict = {"key": key_and_value[0].strip(), "value": key_and_value[1].strip()}
            key_value_list.append(key_value_dict)

    return key_value_list


def remove_empty_values(nested_dict):
    """
    Remove all None values, empty lists, empty strings, and empty dictionaries from a nested JSON dictionary.

    Parameters:
    nested_dict (dict): The nested JSON dictionary from which to remove empty values.

    Returns:
    dict: A new dictionary with all empty values removed.

    This function iterates over each key-value pair in the input dictionary. If a value is a non-empty dictionary,
    list, string, or is not None, it is included in the new dictionary.
    """
    if isinstance(nested_dict, dict):
        cleaned_dict = {
            key: remove_empty_values(value)
            for key, value in nested_dict.items()
            if value is not None
            and not (isinstance(value, list) and len(value) == 0)
            and value != ""
            and remove_empty_values(value) != {}
        }
        return cleaned_dict
    else:
        return nested_dict


def initialize_session_with_nested_data(json_data, child_key, session_key, info_keys):#nested_data, parent_key, child_key, info_keys):
    """
    Initializes the Streamlit session state with data from a nested dictionary structure.

    Parameters:
    nested_data (dict): The nested dictionary containing the data to be initialized.
    parent_key (str): The key in the nested dictionary that contains the parent data.
    child_key (str): The key within the parent data that contains the child data.
    info_keys (dict): A dictionary mapping between child keys of the json data and session state keys.

    The function updates the session state with the count of child data entries and the values
    for each specified key in the child data entries.
    """
    child_data_list = json_data.get(child_key, [])
    st.session_state[session_key] = child_data_list

    st.session_state[child_key + "_count"] = len(child_data_list)
    for index, child_data_entry in enumerate(child_data_list):
        for info_key in info_keys:
            st.session_state[info_keys[info_key] + f"_{index}"] = child_data_entry.get(info_key, "")

def initialize_consideration_data(json_data, child_key):
    """
    Concatenates multiple description strings from a list of dictionaries within nested json data
    into a single newline-separated string and stores it in the Streamlit session state.

    Parameters:
    json_data (dict): The nested json containing the data with descriptions.
    parent_key (str): The key that contains the parent data.
    child_key (str): The key within the parent data that contains the list of dictionaries with descriptions.
    """
    if child_key == "users":

        description_list = [item["description"] for item in json_data]
        joined_descriptions = ", ".join(description_list)
        st.session_state[child_key + "_text"] = joined_descriptions
    else:
        description_list = [item["description"] for item in json_data]
        joined_descriptions = "\n ".join(description_list)
        st.session_state[child_key + "_text"] = joined_descriptions

def navigate_menu():
    """
    Builds the the navigation menu.
    """
    st.sidebar.page_link("home.py", label="🏠 Home")
    st.sidebar.page_link("pages/model_details.py", label="📝 Model Details")
    st.sidebar.page_link("pages/model_parameters.py", label="⚙️ Model Parameters")
    st.sidebar.page_link("pages/considerations.py", label="⚖️ Considerations")
    st.sidebar.page_link("pages/quantitative_analysis.py", label="📊 Quantitative Analysis")
    st.sidebar.page_link("pages/view_model_card.py", label="🔍 View or Download Model Card")