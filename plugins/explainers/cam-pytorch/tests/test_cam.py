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
import torch
import torchvision

torch.manual_seed(0)
import numpy as np

### library to be tested ###
from intel_ai_safety.explainer.cam import pt_cam as cam

###################################

device = torch.device("cpu")


# Non-test, helper functions definitions
def process_output_fasterrcnn(output, class_labels, color, detection_threshold):
    boxes, classes, labels, colors = [], [], [], []
    box = output["boxes"].tolist()
    name = [class_labels[i] for i in output["labels"].detach().numpy()]
    label = output["labels"].detach().numpy()

    for i in range(len(name)):
        score = output["scores"].detach().numpy()[i]
        if score < detection_threshold:
            continue
        boxes.append([int(b) for b in box[i]])
        classes.append(name[i])
        colors.append(color[label[i]])

    return boxes, classes, colors


# Test function definitions
def test_x_gradcam(custom_pyt_CNN):
    model, X_test, class_names, y_test = custom_pyt_CNN
    image = torch.movedim(X_test[0], 0, 2).numpy()
    target_layer = model.conv_layers
    # use the highest-scoring category as the target class
    target_class = None
    image_dims = (28, 28)
    gcam = cam.x_gradcam(model, target_layer, target_class, image, image_dims)
    assert isinstance(gcam, cam.XGradCAM)
    gcam.visualize()


def test_gradcam_pytorch(custom_pyt_CNN):
    model, X_test, class_names, y_test = custom_pyt_CNN
    image = torch.movedim(X_test[0], 0, 2).numpy()
    target_layer = model.conv_layers
    # use the highest-scoring category as the target class
    target_class = None
    image_dims = (28, 28)
    gcam = cam.GradCAM(model, target_layer, image_dims)
    gcam.run_explainer(image, target_class)
    assert isinstance(gcam, cam.XGradCAM)
    gcam.visualize()


def test_eigencam(fasterRCNN, dog_cat_image):
    model, class_labels, color, transform_function = fasterRCNN
    target_layer = model.backbone

    # transform image to fit format of fasterRCNN
    rgb_img = np.float32(dog_cat_image) / 255
    transform = torchvision.transforms.ToTensor()
    input_tensor = transform(rgb_img)
    input_tensor = input_tensor.unsqueeze(0)
    output = model(input_tensor)[0]

    # set the detection threshold
    detection_threshold = 0.9

    # get the box coordinates, classes and colors for the detections
    boxes, classes, colors = process_output_fasterrcnn(output, class_labels, color, detection_threshold)

    ec = cam.eigencam(model, target_layer, boxes, classes, colors, transform_function, dog_cat_image, "cpu")
    assert isinstance(ec, cam.EigenCAM)
    ec.visualize()
