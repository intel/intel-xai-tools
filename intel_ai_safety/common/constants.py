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

from enum import Enum, auto


class ModelFramework(str, Enum):
    """Provide DNN framework constants."""

    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"

    def __str__(self):
        return self.value

    @staticmethod
    def from_str(framework_str):
        enum_dict = dict([(e.value, e) for e in ModelFramework])
        if framework_str in enum_dict:
            return enum_dict[framework_str]
        else:
            raise ValueError(
                "Unsupported model task: {} (Select from: {})".format(framework_str, list(enum_dict.keys()))
            )


class ModelTask(str, Enum):

    def _generate_next_value_(name, start, count, last_values):
        return name.lower()

    CLASSIFICATION = auto()
    MULTICLASSIFICATION = auto()
    MULTILABEL_CLASSIFICATION = auto()
    REGRESSION = auto()
    TEXT_CLASSIFICATION = auto()
    MULTILABEL_TEXT_CLASSIFICATION = auto()
    SENTIMENT_ANALYSIS = auto()
    QUESTION_ANSWERING = auto()
    ENTAILMENT = auto()
    SUMMARIZATIONS = auto()
    IMAGE_CLASSIFICATION = auto()
    MULTILABEL_IMAGE_CLASSIFICATION = auto()
    OBJECT_DETECTION = auto()
    SEMANTIC_SEGMENTATION = auto()
    OTHER = auto()

    def __str__(self):
        return self.value

    def is_image_model(self):
        if self in image_model_tasks:
            return True
        else:
            return False

    def is_text_model(self):
        if self in text_model_tasks:
            return True
        else:
            return False

    def is_classification(self):
        if self in classification_model:
            return True
        else:
            return False

    @staticmethod
    def from_str(model_task_str):
        enum_dict = dict([(e.value, e) for e in ModelTask])
        if model_task_str in enum_dict:
            return enum_dict[model_task_str]
        else:
            raise ValueError(
                "Unsupported model task: {} (Select from: {})".format(model_task_str, list(enum_dict.keys()))
            )


text_model_tasks = {
    ModelTask.TEXT_CLASSIFICATION,
    ModelTask.MULTILABEL_TEXT_CLASSIFICATION,
    ModelTask.SENTIMENT_ANALYSIS,
    ModelTask.QUESTION_ANSWERING,
    ModelTask.ENTAILMENT,
    ModelTask.SUMMARIZATIONS,
}

image_model_tasks = {
    ModelTask.IMAGE_CLASSIFICATION,
    ModelTask.MULTILABEL_IMAGE_CLASSIFICATION,
    ModelTask.OBJECT_DETECTION,
}

classification_model = {
    ModelTask.CLASSIFICATION,
    ModelTask.MULTICLASSIFICATION,
    ModelTask.TEXT_CLASSIFICATION,
    ModelTask.MULTILABEL_TEXT_CLASSIFICATION,
    ModelTask.IMAGE_CLASSIFICATION,
    ModelTask.MULTILABEL_IMAGE_CLASSIFICATION,
}
