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

from .cam import GradCAM

class TFGradCAM(GradCAM):
    '''
    Holds the calculations for the gradient-weighted class activation mapping (gradCAM) of a 
    given image and TensorFlow CNN.

    Args:
      model (tf.keras.functional): the CNN used for classification 
      target_layer (tf.keras.KerasLayer): the convolution layer that you want to analyze (usually the last) 
      dims (tuple of ints): dimension of image (h, w)

        
    Attributes:
      model: the CNN being used
      target_layer: the target convolution being used 
      target_class: the target class being used
      image: the image being used
      gradcam: the result of the gradCAM calculation from the model's target_layer on the image

    Reference:
      https://github.com/ismailuddin/gradcam-tensorflow-2/blob/master/notebooks/GradCam.ipynb 
    '''
    def __init__(self, model, target_layer):
        
        self.model = model
        self.target_layer = target_layer
        
    
    def run_explainer(self, image, target_class):
        '''
        Execute the gradient-based class activation mapping algorithm on target image.

        Args:
          image (numpy.ndarray): image to be analyzed with a shape (h,w,c)
          target_class (int): the index of the target class
        '''

        import numpy as np
        import tensorflow as tf
        import cv2

        self.dims = (image.shape[0], image.shape[1]) 
        self.image = image
        last_conv_layer_model = tf.keras.Model(self.model.inputs, self.target_layer.output)
        classifier_input = tf.keras.Input(shape=self.target_layer.output.shape[1:])
        x = classifier_input
        
        # get the last conv layer and all the proceeding layers  
        last_layers = []
        for layer in reversed(self.model.layers):
            last_layers.append(layer.name)
            if 'conv' in layer.name or 'pool' in layer.name:
                break
        
        # create the classifier model to get the gradient for the
        # target class
        for layer_name in reversed(last_layers):
            x = self.model.get_layer(layer_name)(x)
        classifier_model = tf.keras.Model(classifier_input, x)
        
        with tf.GradientTape() as tape:
            inputs = image[np.newaxis, ...]
            last_conv_layer_output = last_conv_layer_model(inputs)
            tape.watch(last_conv_layer_output)
            preds = classifier_model(last_conv_layer_output)
            top_class_channel = preds[:, target_class]

        grads = tape.gradient(top_class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
        
        last_conv_layer_output = last_conv_layer_output.numpy()[0]
        for i in range(pooled_grads.shape[-1]):
            last_conv_layer_output[:, :, i] *= pooled_grads[i]
        
        # Average over all the filters to get a single 2D array
        gradcam = np.mean(last_conv_layer_output, axis=-1)
        # Clip the values (equivalent to applying ReLU)
        # and then normalise the values
        gradcam = np.clip(gradcam, 0, np.max(gradcam)) / np.max(gradcam)
        self.gradcam = cv2.resize(gradcam, self.dims)
        
    def visualize(self):
        '''
        Plot the gradCAM superimposed on top of the original image.
        '''
        import matplotlib.pyplot as plt
        
        plt.imshow(self.image)
        plt.imshow(self.gradcam, alpha=0.5)
        plt.axis('off')

def tf_gradcam(model, target_layer, target_class, image):
    """
    Generates TFGradCAM object that calculates the gradient-weighted class activation
    mapping of a given image and CNN.

    Args:
      model (tf.keras.Functional): the CNN used for classification 
      target_layer (tf.keras.KerasLayer): the convolution layer that you want to analyze (usually the last) 
      target_class (int): the index of the target class
      image (numpy.ndarray): image to be analyzed with a shape (h,w,c)
      dims (tuple of ints): dimension of image (h, w)

    Returns:
      TFGradCAM

    Reference:
       https://github.com/ismailuddin/gradcam-tensorflow-2/blob/master/notebooks/GradCam.ipynb
    """
    gc = TFGradCAM(model, target_layer)
    gc.run_explainer(image, target_class)
    return gc