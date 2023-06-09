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

import tempfile
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_model_analysis as tfma

TEXT_FEATURE = 'comment_text'
LABEL = 'toxicity'
FEATURE_MAP = {
    LABEL: tf.io.FixedLenFeature([], tf.float32),
    TEXT_FEATURE: tf.io.FixedLenFeature([], tf.string),
    
    'sexual_orientation': tf.io.VarLenFeature(tf.string),
    'gender': tf.io.VarLenFeature(tf.string),
    'religion': tf.io.VarLenFeature(tf.string),
    'race': tf.io.VarLenFeature(tf.string),
    'disability': tf.io.VarLenFeature(tf.string)
}
MODEL_PATH = tempfile.gettempdir()

def get_data_slice(first=1000):
    _, tmp_train = tempfile.mkstemp()
    _, tmp_validate = tempfile.mkstemp()
    dataset_url = 'https://storage.googleapis.com/civil_comments_dataset/'
    train_tf_file = tf.keras.utils.get_file('train_tf_processed.tfrecord',
                                        dataset_url + 'train_tf_processed.tfrecord')
    validate_tf_file = tf.keras.utils.get_file('validate_tf_processed.tfrecord',
                                           dataset_url + 'validate_tf_processed.tfrecord')
    files = zip([train_tf_file, validate_tf_file], [tmp_train, tmp_validate])
    for ds, tmp in files:
        raw_dataset = tf.data.TFRecordDataset(ds)
        writer = tf.data.experimental.TFRecordWriter(tmp)
        writer.write(raw_dataset.take(first))
    return tmp_train, tmp_validate

def train_input_fn():
    def parse_function(serialized):
        # parse_single_example works on tf.train.Example type
        parsed_example = tf.io.parse_single_example(serialized=serialized, features=FEATURE_MAP)
        # fighting the 92%-8% imbalance in the dataset
        # adding `weight` label, doesn't exist already (only FEATURE_MAP keys exist)
        parsed_example['weight'] = tf.add(parsed_example[LABEL], 0.1)  # 0.1 for non-toxic, 1.1 for toxic
        return (parsed_example, parsed_example[LABEL])  # (x, y)
    

    train_dataset = tf.data.TFRecordDataset(filenames=[train_tf_file]).map(parse_function).batch(512)
    return train_dataset

def eval_input_receiver_fn():
    serialized_tf_example = tf.compat.v1.placeholder(dtype=tf.string, shape=[None], name='input_example_placeholder')
    
    receiver_tensors = {'examples': serialized_tf_example}
    features = tf.io.parse_example(serialized_tf_example, FEATURE_MAP)
    features['weight'] = tf.ones_like(features[LABEL])
    
    return tfma.export.EvalInputReceiver(
        features=features,
        receiver_tensors=receiver_tensors,
        labels=features[LABEL]
    )

def build_and_train_model():
    embedded_text_feature_column = hub.text_embedding_column(
        key=TEXT_FEATURE,
        module_spec='https://tfhub.dev/google/nnlm-en-dim128/1')

    classifier = tf.estimator.DNNClassifier(
        hidden_units=[500, 100],
        weight_column='weight',
        feature_columns=[embedded_text_feature_column],
        optimizer=tf.keras.optimizers.legacy.Adagrad(learning_rate=0.003),
        loss_reduction=tf.losses.Reduction.SUM,
        n_classes=2)
    classifier.train(input_fn=train_input_fn, steps=1000)

    return  tfma.export.export_eval_savedmodel(
        estimator = classifier,
        export_dir_base = MODEL_PATH,
        eval_input_receiver_fn = eval_input_receiver_fn
    )

train_tf_file, validate_tf_file = get_data_slice()
