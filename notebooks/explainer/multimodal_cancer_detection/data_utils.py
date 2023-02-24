#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
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

import os
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict

def copy_files_src_to_tgt(samples, fns_dict, src_folder, tgt_folder):
    '''Copies the image files from the original dataset to the target, grouped
    dataset folder.

    '''
    for sample in samples:
        files_to_copy = fns_dict.get(sample)
        for _file in files_to_copy:
            src_fn = os.path.join(src_folder, _file)
            tgt_fn = os.path.join(tgt_folder, _file)
            shutil.copy2(src_fn, tgt_fn)

def split_images(src_folder, tgt_folder):
    '''Splits the original image dataset in the src_folder such
    that no PID is in both train and test subsets in the tgt_folder.

    '''
    labels = os.listdir(src_folder)
    print("Number of labels = ", len(labels))
    print("Labels are: \n", labels)
    for label in labels:
        fns = os.listdir(os.path.join(src_folder, label))
        fns.sort()
        fns_root = ['_'.join(x.split('_')[:2]) for x in fns]
        # Convert list of tuples to dictionary value lists
        print("\nCreating default dict for stratifying the subject in {}.".format(label))
        fns_dict = defaultdict(list)
        for i, j in zip(fns_root, fns):
            fns_dict[i].append(j)
        train_samples, test_samples = train_test_split(list(fns_dict.keys()), test_size=0.2, random_state=100)

        src_dir = os.path.join(src_folder, label)
        tgt_dir = os.path.join(tgt_folder, 'train', label)
        os.makedirs(tgt_dir, exist_ok=True)
        copy_files_src_to_tgt(train_samples, fns_dict, src_dir, tgt_dir)

        tgt_dir = os.path.join(tgt_folder, 'test', label)
        os.makedirs(tgt_dir, exist_ok=True)
        copy_files_src_to_tgt(test_samples, fns_dict, src_dir, tgt_dir)

        print("Done splitting the files for label = {}\n".format(label))
    print("Done splitting the data. Output data is here: ", tgt_folder)


def get_subject_id(image_name):
    '''Returns the PID from the image_name.'''

    image_name = image_name.split("/")[-1]
    patient_id = "".join(image_name.split("_")[:2])[1:]
    return patient_id


def create_patient_id_list(train_image_data_folder, test_image_data_folder, folder):
    '''returns the list of PIDs in order that aligned with 
    the image dataset.

    '''
    train_folder_pth = os.path.join(folder, train_image_data_folder)
    test_folder_pth = os.path.join(folder, test_image_data_folder)

    train_patient_id_list = []
    test_patient_id_list = []

    for fldr in os.listdir(train_folder_pth):
        for f in os.listdir(os.path.join(train_folder_pth, fldr)):
            train_patient_id_list.append(get_subject_id(f))

    for fldr in os.listdir(test_folder_pth):
        for f in os.listdir(os.path.join(test_folder_pth, fldr)):
            test_patient_id_list.append(get_subject_id(f))
    
    return np.unique(train_patient_id_list), np.unique(test_patient_id_list)


def read_annotation_file(
    folder,
    file_name,
    label_column,
    data_column,
    patient_id,
    patient_id_list,
    image_data_folder
):
    '''Creates a pandas DataFrame from the csv file_name with 
    a label_column, data_column and a patient_id column and orders 
    the text examples in alignment with the image dataset.
    Returns the pandas DataFrame, the map and reverse map of the labels.

    '''
    df_path = os.path.join(folder, file_name)
    df = pd.read_csv(df_path)
    label_map, reverse_label_map = label2map(df, label_column)

    if patient_id_list is not None:
        df = df[df[patient_id].isin(patient_id_list)]
    else:
        image_name_list = []
        for label in os.listdir(image_data_folder):
            image_name_list.extend(os.listdir(os.path.join(image_data_folder, label)))
        df = df[df[patient_id].isin(np.unique([get_subject_id(i) for i in image_name_list]))]

    df_new = pd.DataFrame(columns=[label_column, data_column, patient_id])
    for i in df[patient_id].unique():
        annotation = " ".join(df[df[patient_id].isin([i])][data_column].to_list())
        temp_labels = df[df[patient_id] == i][label_column].unique()
        if len(temp_labels) == 1:
            df_new.loc[len(df_new)] = [temp_labels[0], annotation, i]
        else:
            if patient_id_list is not None:
                # this is the case only shows for inference
                # label assigne as a place holder
                df_new.loc[len(df_new)] = ["Normal", annotation, i]
            else:
                Warning("Conflict in labelling ....")

    return df_new, label_map, reverse_label_map


def label2map(df, label_column):
    '''Creates and returns the dictionaries holding the label and 
    reverse label maps.

    '''
    label_map, reverse_label_map = {}, {}
    for i, v in enumerate(df[label_column].unique().tolist()):
        label_map[v] = i
        reverse_label_map[i] = v

    return label_map, reverse_label_map


def create_train_test_set(df, patient_id, train_patient_id_list, test_patient_id_list):
    '''Splits the DataFrame df into a training and
    testing DataFrame and returns them.

    '''
    '''
    train_label, test_label = train_test_split(
        patient_id_list, test_size=0.33, random_state=42
    )
    '''

    df_test = df[df[patient_id].isin(test_patient_id_list)]
    df_train = df[df[patient_id].isin(train_patient_id_list)]

    return df_train, df_test


def split_annotation(folder, file_name, train_image_data_folder, test_image_data_folder):
    '''Restructures the original csv file_name such that each PID's
    collection of text entries is concatenated together as one example.
    This is done so that PIDs are not seen in both training and testing.
    Returns a DataFrame with the grouped examples.

    '''
    label_column = "label"
    data_column = "symptoms"
    patient_id = "Patient_ID"
    patient_id_list = None

    train_df, label_map, reverse_label_map = read_annotation_file(
        folder,
        file_name,
        label_column,
        data_column,
        patient_id,
        patient_id_list,
        train_image_data_folder
    )
    test_df, label_map, reverse_label_map = read_annotation_file(
        folder,
        file_name,
        label_column,
        data_column,
        patient_id,
        patient_id_list,
        test_image_data_folder
    )
    df = pd.concat([train_df, test_df])

    train_patient_id_list, test_patient_id_list = create_patient_id_list(train_image_data_folder, test_image_data_folder, folder)
    df_train, df_test = create_train_test_set(df, patient_id, train_patient_id_list, test_patient_id_list)
    
    return df_train, df_test 
