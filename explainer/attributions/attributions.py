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

import pandas as pd
import numpy as np
from typing import Union, Optional, Callable, List
from .attributions_info import force_plot_info_panel
from explainer.utils.graphics.info import InfoPanel
from explainer.utils.types import TorchTensor
from explainer.utils.model.model_framework import (is_tf_model,
                                                   is_pt_model,
                                                   raise_unknown_model_error)


class FeatureAttributions:
    '''
    Attributions base class. Holds the shap API.
    '''
    def __init__(self):
        import shap
        shap.initjs()
        self.shap = shap
        self.datasets = shap.datasets
        self.plots = self.shap.plots
        self.bar_plot = self.plots.bar
        self.image_plot = self.plots.image
        self.force_plot = self.shap.force_plot
        self.text_plot = self.plots.text
        self.waterfall_plot = self.shap.waterfall_plot
        self.info_panel = {}

    def __call__(self, *args, **kwargs):
        pass

    def visualize(self, data):
        pass

    def get_info(self):
        """Display into panel in Jupyter Enviornment"""
        if self.info_panel:
            info = InfoPanel(**self.info_panel)
            info.show()

class DeepExplainer(FeatureAttributions):
    '''
    Approximate conditional expectations of shap values for deep learning models using a variation of the DeepLIFT algorithm
     (Shrikumar, Greenside, and Kundaje, arXiv 2017)

    Args:
      model (tf.keras.functional or pytorch.nn.Module): CNN model to be interpreted
      background_images (numpy.ndarray, pandas.DataFrame or torch.tensor): the selection of background images used to integrate output features
        across each target image
      targetImages (numpy.ndarray, pandas.DataFrame or torch.tensor): the images to be interpreted
      labels (list of strings): list of label names for the given classification problem

    Attributes:
      target_images: images used to explain predictions
      explainer: the shap DeepExplainer object
      shap_values: the resulting shap value estimations on the target images
      labels: the class labels of the given classification problem 
    
    Reference:
      https://shap-lrjball.readthedocs.io/en/latest/generated/shap.DeepExplainer.html
    '''
    def __new__(cls, model, *args):    
        if is_tf_model(model):
            # do nothing 
            return super().__new__(cls)
        elif is_pt_model(model):
            from .pt_attributions import PTDeepExplainer
            return super().__new__(PTDeepExplainer)
        else:
            raise_unknown_model_error()

    def __init__(self, 
                 model, 
                 background_images: Union[np.ndarray,  pd.DataFrame, TorchTensor], 
                 target_images: Union[np.ndarray,  pd.DataFrame, TorchTensor], 
                 labels: Union[List[str], np.ndarray]
                 ) -> None:
        super().__init__()
        self.target_images = target_images
        self.explainer = self.shap.DeepExplainer(model, background_images)
        self.shap_values = self.explainer.shap_values(target_images)
        self.labels = labels

    

    def visualize(self) -> None:
        '''
        plot superposition of shap estimations on original image(s) across all labels predictions
        '''

        self.shap.image_plot(
            self.shap_values,
            self.target_images,
            np.array([list(self.labels)]*len(self.target_images)),
        )


class GradientExplainer(FeatureAttributions):
    '''
    Approximate expected gradients of differentiable models using a variation of the integrated gradients algorithm
     (Sundararajan et al. 2017)

    Args:
      model (tf.keras.functional or pytorch.nn.Module): CNN model to be interpreted
      background_images (numpy.ndarray, pandas.DataFrame or torch.tensor): the selection of background images 
        used to integrate output features across each target image
      target_images (numpy.ndarray, pandas.DataFrame or torch.tensor): the images to be interpreted
      labels (list of strings): list of label names for the given classification problem
      ranked_outputs (int): the number of top label predictions to be analyzed. Defaults to 1.

    Attributes:
      target_images: images used to explain predictions
      ranked_outputs: the number of top label predictions
      explainer: the shap GradientExplainer object
      shap_values: the resulting shap value estimations on the target images
      indexes: indexes where for the corresponding rankings of the each target image ranking
      labels: the class labels of the given classification problem 

    Reference:
      https://shap-lrjball.readthedocs.io/en/latest/generated/shap.GradientExplainer.html
    '''

    def __new__(cls, model, *args):    
        if is_tf_model(model):
            # do nothing
            return super().__new__(cls)
        elif is_pt_model(model):
            from .pt_attributions import PTGradientExplainer
            return super().__new__(PTGradientExplainer)
        else:
            raise_unknown_model_error()

    def __init__(self, 
                 model, 
                 background_images: Union[np.ndarray, pd.DataFrame, TorchTensor], 
                 target_images: Union[np.ndarray, pd.DataFrame, TorchTensor], 
                 labels: Union[List[str], np.ndarray],
                 ranked_outputs: Optional[int] = 1,
                 ) -> None:
        super().__init__()
        self.target_images = target_images
        self.ranked_outputs = ranked_outputs
        self.labels = labels
        self.explainer = self.shap.GradientExplainer(model, background_images)
        self.shap_values, self.indexes = self.explainer.shap_values(self.target_images, ranked_outputs=ranked_outputs)

        if self.indexes.shape == (1,1):
            self.idxs_to_plot = [self.labels[self.indexes[0][0]]]
        else:
            self.idxs_to_plot = np.array(self.labels)[self.indexes]

    def visualize(self) -> None:
        '''
        plot superposition of shap estimations on original image(s) across top ranked_outputs 
        '''
        self.shap.image_plot(
            self.shap_values, 
            self.target_images, 
            self.idxs_to_plot
        )


class KernelExplainer(FeatureAttributions):
    '''
    Approximate shap values via a combination of local interpretable model-agnostic explanations (LIME) and
    a weighted linear regression.

    Args:
      model (function): "black box" prediction function that takes an input array of shape (n samples, m features)
      and outputs an array of n predictions.
      background (numpy.ndarray or pandas.DataFrame): the selection of background examples used to integrate output features
        across each target example
      targets (numpy.ndarray or pandas.DataFrame): the target examples to be interpreted
      nsamples (int): the number of times to re-evaluate the model per prediction. Defaults to 64.

    Attributes:
      bg: background examples
      targets: target examples
      explainer: shap KernelExplainer object
      shap_values: the resulting shap value estimations on the target examples 

    Reference:
    https://shap-lrjball.readthedocs.io/en/latest/generated/shap.KernelExplainer.html
    '''
    def __init__(self, 
                 model: Callable[[np.ndarray], np.ndarray], 
                 background: Union[np.ndarray, pd.DataFrame],
                 targets: Union[np.ndarray, pd.DataFrame], 
                 nsamples: int = 64
                 ) -> None:
        super().__init__()
        self.bg = background
        self.targets = targets
        self.explainer = self.shap.KernelExplainer(model, self.bg)
        self.shap_values = self.explainer.shap_values(self.targets, nsamples=nsamples)
        self.info_panel = force_plot_info_panel

    def visualize(self):
        '''
        Display the force plot of the of the target example(s)
        '''
        return self.force_plot(self.explainer.expected_value, self.shap_values[0], self.targets)


class PartitionExplainer(FeatureAttributions):
    '''
    Approximate an extension of shap values, known as Owen values, by recursively computing shap values 
    through a hierarchy of features that define feature coalitions. This is the base partition explainer class
    that generates partition explainer children objects for text or image classification.

    Args:
      task_type (string): 'text' or 'image' to choose which classification domain to be explained
      *args: the remaining arguments required for child class instantiation
    '''
    def __new__(cls, task_type: str, *args) -> None:
        if task_type == 'text':
            return super().__new__(PartitionTextExplainer)
        elif task_type == 'image':
            return super().__new__(PartitionImageExplainer)
        else:
            raise ValueError(f"Task type {type(task_type)} is unsupported: please use 'text' or 'image'")
    


class PartitionImageExplainer(PartitionExplainer, FeatureAttributions):
    '''
    Image classification-based partition explanation. 
    
    Args:
      task_type (string): 'text' or 'image' used in PartitionExplainer generator class. It is not used in this 
      child class.
      model (function): "black box" prediction function that takes an input array of shape (n samples, m features)
      and outputs an array of n predictions.
      labels (list): list of label names (strings) for the given classification problem
      shape (tuple): height (int) and width (int) of images to be analyzed

    Attributes:
      explainer: shap PartitionExplainer object
      shap_values: the resulting shap value estimations on the target images 
    '''
    def __init__(self, 
                 task_type: str, 
                 model: Callable[[np.ndarray], np.ndarray], 
                 labels: List[str], 
                 shape: tuple[int, int] 
                 ) -> None:
        FeatureAttributions.__init__(self)
        PartitionExplainer.__init__(self)
        self.shap_values = None
        self.explainer = self.shap.Explainer(model, self.shap.maskers.Image('inpaint_telea', shape), output_names=labels)      

    def run_explainer(self, 
                      target_images: np.ndarray, 
                      top_n: int = 1, 
                      max_evals: int = 64
                      ) -> None:
        '''
        Execute the partition explanation on the target_images.

        Args:
          target_images (numpy.ndarray): n images in the shape (n, height, width, channels)
          in a better the estimation. Defaults to 64. 
          top_n (int): gather shap values for the top n most probable classes per image. Defaults to 1.
          max_evals (int): number of evaluations used in the shap estimation. The higher the number result 

        Returns:
          None
        '''
        self.shap_values = self.explainer(target_images, max_evals=max_evals, outputs=self.shap.Explanation.argsort.flip[:top_n])
    
    def visualize(self) -> None:
        '''
        Plot superposition of shap estimations on original image(s) across top ranked_outputs 
        '''
        self.image_plot(self.shap_values)

class PartitionTextExplainer(PartitionExplainer, FeatureAttributions):
    '''
    Text classification-based partition explanation (using HuggingFace Transformers API). 

    Args:
      task_type (string): 'text' or 'image' used in PartitionExplainer generator class. It is not used in this 
      child class.
      model (function): "black box" prediction function that takes an input array of shape (n samples, m features)
      and outputs an array of n predictions.
      labels (list): list of label names (strings) for the given classification problem
      tokenizer (string or callable): tokenizer used to break apart strings during masking of the text

    Attributes:
      explainer: shap PartitionExplainer object
      shap_values: the resulting shap value estimations on the target examples 

    Reference:
    https://shap-lrjball.readthedocs.io/en/latest/generated/shap.PartitionExplainer.html 
    '''
    def __init__(self, 
                 task_type: str, 
                 model: Callable[[np.ndarray], np.ndarray], 
                 labels: List[str], 
                 tokenizer: Union[Callable[[str], str], str]) -> None:

        FeatureAttributions.__init__(self)
        PartitionExplainer.__init__(self)
        self.shap_values = None
        self.explainer = self.shap.Explainer(model, masker=self.shap.maskers.Text(tokenizer=tokenizer), output_names=labels)

    def run_explainer(self, 
                      target_text: str, 
                      max_evals: int = 64) -> None:
        '''
        Execute the partition explanation on the target_text.

        Args:
          target_text (numpy.ndarray): 1-d numpy array of strings holding the text examples
          max_evals (int): number of evaluations used in the shap estimation. The higher the number result 
          in a better the estimation. Defaults to 64. 

        Returns:
          None
        '''
        self.shap_values = self.explainer(target_text, max_evals=max_evals)

    def visualize(self):
        '''
        Display the force plot of the of the target example(s)
        '''
        self.text_plot(self.shap_values)


class PipelineExplainer(FeatureAttributions):
    def __init__(self, task, model):
        import transformers

        super().__init__(task)
        self.pipeline = transformers.pipeline(
            task, return_all_scores=True
        )
        self.pipeline.model = model

    def __call__(self, *args, **kwargs):
        self.explainer = self.shap.Explainer(self.pipeline)
        data = args[0]
        self.pipeline(data)
        self.shap_values = self.explainer(data)
        return self

    def visualize(self, label):
        self.shap.plots.text(self.shap_values[:, :, label])




def explainer():
    """
    Calls FeatureAttributions
    Returns FeatureAttributions which has native references as attributes

    Returns:
      FeatureAttributions
    
    Example:
      >>> from explainer import attributions
      >>> explainer = attributions.explainer()
      <IPython.core.display.HTML object>
      >>> explainer.shap.__version__
      '0.41.0' 
    """
    return FeatureAttributions()


def kernel_explainer(model, background, targets, nsamples=500):
    """
    Returns a SHAP KernelExplainer, using the Kernel SHAP method
    to explain the output of any function.

    Args:
      model: model
      data: dataframe

    Returns:
      KernelExplainer

    Reference:
      https://shap-lrjball.readthedocs.io/en/latest/generated/shap.KernelExplainer.html
    """
    return KernelExplainer(model, background, targets, nsamples=nsamples)


def deep_explainer(model, backgroundImages, targetImages, labels) -> DeepExplainer:
    """
    Returns a SHAP DeepExplainer that approximate SHAP values for deep learning models.

    Args:
      model: model
      backgroundImages: list
      targetImages: list
      labels: list

    Returns:
      DeepExplainer

    Reference:
      https://shap-lrjball.readthedocs.io/en/latest/generated/shap.DeepExplainer.html
    """
    return DeepExplainer(model, backgroundImages, targetImages, labels)


def gradient_explainer(model, 
                       background_images: Union[np.ndarray,  pd.DataFrame, TorchTensor],
                       target_images: Union[np.ndarray,  pd.DataFrame, TorchTensor], 
                       labels: Union[List[str], np.ndarray],
                       ranked_outputs: Optional[int] = 1,
                 ) -> GradientExplainer:
    """
    Sets up a SHAP GradientExplainer, explains a model using expected gradients.

    Args:
      model: model
      background_images: list
      target_images: list
      labels: list
      ranked_outputs: int (options and defaults to 1)

    Returns:
      GradientExplainer

    Reference:
      https://shap-lrjball.readthedocs.io/en/latest/generated/shap.GradientExplainer.html
    """
    return GradientExplainer(model, background_images, target_images, labels, ranked_outputs)


def partition_text_explainer(model, labels, target_text, tokenizer, max_evals=64):
    """
    Instantiates PartitionExplainer for text classification and executes run_explainer()

    Args:
      model (function): the model
      tokenizer (string or callable): the tokens you want to mask
      categories (list): the category names

    Reference:
      https://shap-lrjball.readthedocs.io/en/latest/generated/shap.PartitionExplainer.html

    Returns:
      PartitionExplainer
    """
    pe = PartitionExplainer('text', model, labels, tokenizer)
    pe.run_explainer(target_text, max_evals=max_evals)
    return pe


def partition_image_explainer(model, labels, target_images, top_n=1, max_evals=64):
    """
    Instantiates PartitionExplainer for image classification and executes run_explainer()

    Args:
      model (function): the model
      tokenizer (string or callable): the tokens you want to mask
      categories (list): the category names

    Reference:
      https://shap-lrjball.readthedocs.io/en/latest/generated/shap.PartitionExplainer.html

    Returns:
      PartitionImageExplainer
    """
    pe = PartitionExplainer('image', model, labels, target_images[0].shape)
    pe.run_explainer(target_images, top_n=top_n, max_evals=max_evals)
    return pe

def zero_shot(pipe, text):
    print(f"Shap version used: {shap.__version__}")
    explainer = shap.Explainer(pipe)
    shap_values = explainer(text)
    prediction = pipe(text)
    print(f"Model predictions are: {prediction}")
    shap.plots.text(shap_values)
    # Let's visualize the feature importance towards the outcome - sports
    shap.plots.bar(shap_values[0, :, "sports"])


def sentiment_analysis(model, data):
    """
    Uses HuggingFace pipeline to instantiate a model for task 'sentiment-analysis'
    Returns PipelineExplainer

    Args:
      model: model
      data: examples used for sentiment-analysis

    Returns:
      PipelineExplainer
    """
    return PipelineExplainer('sentiment-analysis', model)(data)
