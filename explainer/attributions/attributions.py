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

import torch
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
    def __init__(self, model, background_images, target_images, labels):
        super().__init__()
        self.target_images = target_images
        self.explainer = self.shap.DeepExplainer(model, background_images)
        self.shap_values = self.explainer.shap_values(target_images)
        self.labels = labels

    def visualize(self):
        import numpy as np
        import torch
      
        arr = np.full((len(self.labels)), " ")
        if torch.is_tensor(self.target_images):
            self.shap_values = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in self.shap_values]
            self.target_images = -np.swapaxes(np.swapaxes(self.target_images.numpy(), 1, -1), 1, 2)

        self.shap.image_plot(
            self.shap_values,
            self.target_images,
            np.array([list(self.labels)]*len(self.target_images)),
        )


class GradientExplainer(FeatureAttributions):
    def __init__(self, model, background_images, target_images, ranked_outputs, labels ):
        import numpy as np
        super().__init__()
        self.target_images = target_images
        self.ranked_outputs = ranked_outputs
        self.labels = labels
        self.explainer = self.shap.GradientExplainer(model, background_images)
        self.shap_values, self.indexes = self.explainer.shap_values(self.target_images, ranked_outputs=ranked_outputs)

    def visualize(self):
        import numpy as np
        import torch

        if torch.is_tensor(self.target_images):
            self.shap_values = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in self.shap_values]
            self.target_images = -np.swapaxes(np.swapaxes(self.target_images.numpy(), 1, -1), 1, 2) 

        # check if 
        if self.ranked_outputs == 1 and len(self.labels[self.indexes]) == 1:
            idxs_to_plot = np.expand_dims(np.expand_dims(self.labels[self.indexes], 0), 0)
        else:
            idxs_to_plot = self.labels[self.indexes]

        self.shap.image_plot(
            self.shap_values, 
            self.target_images, 
            idxs_to_plot
        )


class KernelExplainer(FeatureAttributions):
    def __init__(self, model, background, targets, nsamples):
        super().__init__()
        self.bg = background
        self.targets = targets
        self.explainer = self.shap.KernelExplainer(model, self.bg)
        self.shap_values = self.explainer.shap_values(self.targets, nsamples=nsamples)
        self.info_panel = force_plot_info_panel

    def visualize(self):
        return self.force_plot(self.explainer.expected_value, self.shap_values[0], self.targets)


class PartitionExplainer(FeatureAttributions):
    def __init__(self, model, tokenizer, categories):
        super().__init__()
        self.masker = self.shap.maskers.Text(tokenizer=tokenizer)
        self.explainer = self.shap.Explainer(model, masker=self.masker, output_names=categories)

    def __call__(self, *args, **kwargs):
        data = args[0]
        self.shap_values = self.explainer(data)
        return self

    def visualize(self):
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


class Captum_Saliency:
    def __init__(self, model):
        from captum.attr import Saliency

        self.saliency = Saliency(model)

    def visualize(self, input, labels, original_image, imageTitle):
        import numpy as np
        from captum.attr import visualization as viz

        self.grads = self.saliency.attribute(input, target=labels.item())
        self.grads = np.transpose(
            self.grads.squeeze().cpu().detach().numpy(), (1, 2, 0)
        )
        viz.visualize_image_attr(
            self.grads,
            original_image,
            method="blended_heat_map",
            sign="absolute_value",
            show_colorbar=True,
            title=imageTitle,
        )


class Captum_IntegratedGradients:
    def __init__(self, model):
        from captum.attr import IntegratedGradients

        self.ig = IntegratedGradients(model)

    def visualize(self, input, labels, original_image, imageTitle):
        import numpy as np
        from captum.attr import visualization as viz

        self.attr_ig = self.ig.attribute(input, target=labels, baselines=input * 0)
        self.attr_ig = np.transpose(
            self.attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)
        )
        viz.visualize_image_attr(
            self.attr_ig,
            original_image,
            method="blended_heat_map",
            sign="all",
            show_colorbar=True,
            title=imageTitle,
        )


class Captum_DeepLift:
    def __init__(self, model):
        from captum.attr import DeepLift

        self.dl = DeepLift(model)

    def visualize(self, input, labels, original_image, imageTitle):
        import numpy as np
        from captum.attr import visualization as viz

        self.attr_dl = self.dl.attribute(input, target=labels, baselines=input * 0)
        self.attr_dl = np.transpose(
            self.attr_dl.squeeze(0).cpu().detach().numpy(), (1, 2, 0)
        )
        viz.visualize_image_attr(
            self.attr_dl,
            original_image,
            method="blended_heat_map",
            sign="all",
            show_colorbar=True,
            title=imageTitle,
        )


class Captum_SmoothGrad:
    def __init__(self, model):
        from captum.attr import IntegratedGradients
        from captum.attr import NoiseTunnel

        self.ig = IntegratedGradients(model)
        self.nt = NoiseTunnel(self.ig)

    def visualize(self, input, labels, original_image, imageTitle):
        import numpy as np
        from captum.attr import visualization as viz

        self.attr_ig_nt = self.nt.attribute(
            input,
            target=labels,
            baselines=input * 0,
            nt_type="smoothgrad_sq",
            nt_samples=100,
            stdevs=0.2,
        )
        self.attr_ig_nt = np.transpose(
            self.attr_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0)
        )
        viz.visualize_image_attr(
            self.attr_ig_nt,
            original_image,
            method="blended_heat_map",
            sign="absolute_value",
            outlier_perc=10,
            show_colorbar=True,
            title=imageTitle,
        )


class Captum_FeatureAblation:
    def __init__(self, model):
        from captum.attr import FeatureAblation

        self.ablator = FeatureAblation(model)

    def visualize(self, input, labels, original_image, imageTitle):
        import numpy as np
        from captum.attr import visualization as viz

        self.fa_attr = self.ablator.attribute(
            input, target=labels, baselines=input * 0
        )
        self.fa_attr = np.transpose(
            self.fa_attr.squeeze(0).cpu().detach().numpy(), (1, 2, 0)
        )
        viz.visualize_image_attr(
            self.fa_attr,
            original_image,
            method="blended_heat_map",
            sign="all",
            show_colorbar=True,
            title=imageTitle,
        )

def explainer():
    """
    Calls FeatureAttributions
    Returns FeatureAttributions which has native references as attributes

    Returns:
      FeatureAttributions
    
    Example:
      >>> from explainer.explainers import feature_attributions_explainer
      >>> explainer = feature_attributions_explainer.explainer
      >>> ??explainer
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


def deep_explainer(model, backgroundImages, targetImages, labels):
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


def gradient_explainer(model, backgroundImages, targetImages, rankedOutputs, labels):
    """
    Sets up a SHAP GradientExplainer, explains a model using expected gradients.

    Args:
      model: model
      backgroundImages: list
      targetImages: list
      rankedOutputs: int
      labels: list

    Returns:
      GradientExplainer

    Reference:
      https://shap-lrjball.readthedocs.io/en/latest/generated/shap.GradientExplainer.html
    """
    return GradientExplainer(model, backgroundImages, targetImages, rankedOutputs, labels)


def partition_explainer(model, tokenizer, categories):
    """
    Calls PartitionExplainer with the model, masker and categories
    Returns a SHAP PartitionExplainer that explain the output of any function.

    Args:
      model (function): the model
      tokenizer (str): the tokens you want to mask
      categories (list): the category names

    Reference:
      https://shap-lrjball.readthedocs.io/en/latest/generated/shap.PartitionExplainer.html

    Returns:
      PartitionExplainer
    
    Example:
      >>> def make_pred(X_batch_text):
      >>>   X_batch = vectorizer.transform(X_batch_text).toarray()
      >>>   preds = model.predict(X_batch)
      >>>   return preds
      >>>
      >>> from explainer.explainers import feature_attributions_explainer
      >>> partition_explainer = feature_attributions_explainer.partitionexplainer
      >>> partition_explainer = partitione_explainer(make_pred, r"\W+",selected_categories)(X_batch_text)
      >>> partition_explainer.visualize()
    """
    return PartitionExplainer(model, tokenizer, categories)


def saliency(model):
    """
    Returns a Captum Saliency. This provides a baseline approach for computing
    input attribution, returning the gradients with respect to inputs.

    Args:
      model: pytorch model

    Returns:
      captum.attr.Saliency

    Reference:
      https://captum.ai/api/saliency.html
    """
    return Captum_Saliency(model)


def integratedgradients(model):
    """
    Returns a Captum IntegratedGradients

    Args:
      model: pytorch model

    Returns:
      captum.attr.IntegratedGradients

    Reference:
      https://captum.ai/api/integrated_gradients.html
    """
    return Captum_IntegratedGradients(model)


def deeplift(model):
    """
    Returns a Captum DeepLift

    Args:
      model: pytorch model

    Returns:
      captum.attr.DeepLift

    Reference:
      https://captum.ai/api/deep_lift.html
    """
    return Captum_DeepLift(model)


def smoothgrad(model):
    """
    Returns a Captum Integrated Gradients, Noise Tunnel SmoothGrad

    Args:
      model: pytorch model

    Returns:
      captum.attr.NoiseTunnel

    Reference:
      https://captum.ai/api/integrated_gradients.html
      https://captum.ai/api/noise_tunnel.html
    """
    return Captum_SmoothGrad(model)


def featureablation(model):
    """
    Returns a Captum FeatureAblation

    Args:
      model: pytorch model

    Returns:
      captum.attr.FeatureAblation

    Reference:
      https://captum.ai/api/feature_ablation.html
    """
    return Captum_FeatureAblation(model)


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

    Example:
      >>> from explainer.api import ExplainerContextManager as ec
      >>> with ec('feature_attributions_explainer') as fe:
      >>>   fe.sentiment_analysis(model.model, raw_text_input).visualize(1)
    """
    return PipelineExplainer('sentiment-analysis', model)(data)
