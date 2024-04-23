#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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
### libraries to support tests ###
import pytest
import numpy as np
### library to be tested ###
from intel_ai_safety.explainer.cam import tf_cam as cam
###################################


# Non-test, helper functions definitions
def process_output_fasterrcnn(output, class_labels, color, detection_threshold):
        boxes, classes, labels, colors = [], [], [], []
        box = output['boxes'].tolist()
        name = [class_labels[i] for i in output['labels'].detach().numpy()]
        label = output['labels'].detach().numpy()

        for i in range(len(name)):
            score = output['scores'].detach().numpy()[i]
            if score < detection_threshold:
                continue
            boxes.append([int(b) for b in box[i]])
            classes.append(name[i])
            colors.append(color[label[i]])

        return boxes, classes, colors


def test_tf_gradcam_vgg(tf_VGG, dog_cat_image):
    target_class = 281
    target_layer = tf_VGG.get_layer('block5_conv3')
    gcam = cam.tf_gradcam(tf_VGG, target_layer, target_class, dog_cat_image)
    assert isinstance(gcam, cam.TFGradCAM)
    gcam.visualize()


def test_tf_gradcam_resnet50(tf_resnet50, dog_cat_image):
    target_class = 281
    target_layer = tf_resnet50.get_layer('conv5_block3_out')
    gcam = cam.tf_gradcam(tf_resnet50, target_layer, target_class, dog_cat_image)
    assert isinstance(gcam, cam.TFGradCAM)
    gcam.visualize()


def test_gradcam_tf_resnet50(tf_resnet50, dog_cat_image):
    target_class = 281
    target_layer = tf_resnet50.get_layer('conv5_block3_out')
    gcam = cam.GradCAM(tf_resnet50, target_layer)
    gcam.run_explainer(dog_cat_image, target_class)
    assert isinstance(gcam, cam.TFGradCAM)
    gcam.visualize()
