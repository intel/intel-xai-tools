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

class FeatureAttributions:
    '''
    Attributions base class. Holds the SHAP API.
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

    def __call__(self, *args, **kwargs):
        pass

    def visualize(self, data):
        pass


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
    
    Methods:
      visualize: superimpose SHAP estimations on top of original image(s) across all labels predictions

    Reference:
      https://shap-lrjball.readthedocs.io/en/latest/generated/shap.DeepExplainer.html
    '''
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
    '''
    Approximate expected gradients of differentiable models using a variation of the integrated gradients algorithm
     (Sundararajan et al. 2017)

    Args:
      model (tf.keras.functional or pytorch.nn.Module): CNN model to be interpreted
      background_images (numpy.ndarray, pandas.DataFrame or torch.tensor): the selection of background images 
        used to integrate output features across each target image
      target_images (numpy.ndarray, pandas.DataFrame or torch.tensor): the images to be interpreted
      ranked_outputs (int): the number of top label predictions to be analyzed
      labels (list of strings): list of label names for the given classification problem

    Attributes:
      target_images: images used to explain predictions
      ranked_outputs: the number of top label predictions
      explainer: the shap GradientExplainer object
      shap_values: the resulting shap value estimations on the target images
      indexes: indexes where for the corresponding rankings of the each target image ranking
      labels: the class labels of the given classification problem 


    Methods:
      visualize: superimpose expected gradients estimations on top of original image(s) across the top ranked_outputs predictions

    Reference:
      https://shap-lrjball.readthedocs.io/en/latest/generated/shap.GradientExplainer.html
    '''
    def __init__(self, model, background_images, target_images, ranked_outputs, labels):
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
    '''
    Approximate shap values via a combination of local interpretable model-agnostic explanations (LIME) and
    a weighted linear regression.

    Args:
      model (function): "black box" prediction function that takes an input array of shape (n samples, m features)
      and outputs an array of n predictions.
      background (numpy.ndarray or pandas.DataFrame): the selection of background examples used to integrate output features
        across each target example
      targets (numpy.ndarray or pandas.DataFrame): the target examples to be interpreted
      ranked_outputs (int): the number of top label predictions to be analyzed
      labels (list of strings): list of label names for the given classification problem

    Attributes:
      bg: background examples
      targets: target examples
      explainer: shap KernelExplainer object
      shap_values: the resulting shap value estimations on the target examples 

    Methods:
      visualize: display the force plot of the of the target example(s)

    Reference:
    https://shap-lrjball.readthedocs.io/en/latest/generated/shap.KernelExplainer.html
    '''
    def __init__(self, model, background, targets, nsamples):
        super().__init__()
        self.bg = background
        self.targets = targets
        self.explainer = self.shap.KernelExplainer(model, self.bg)
        self.shap_values = self.explainer.shap_values(self.targets, nsamples=nsamples)

    def visualize(self):
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
    def __new__(cls, task_type, *args):
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
    def __init__(self, task_type, model, labels, shape):
        FeatureAttributions.__init__(self)
        PartitionExplainer.__init__(self)
        self.shap_values = None
        self.explainer = self.shap.Explainer(model, self.shap.maskers.Image('inpaint_telea', shape), output_names=labels)      

    def run_explainer(self, target_images, top_n=1, max_evals=64):
        '''
        Execute the partition explanation on the target_images.

        Args:
          target_images (numpy.ndarray): n images in the shape (n, height, width, channels)
          max_evals (int): number of evaluations used in the shap estimation. The higher the number result 
          in a better the estimation. Defaults to 64. 
          top_n (int): gather shap values for the top n most probable classes per image. Defaults to 1.

        Returns:
          None
        '''
        self.shap_values = self.explainer(target_images, max_evals=max_evals, outputs=self.shap.Explanation.argsort.flip[:top_n])
    
    def visualize(self):
        self.image_plot(self.shap_values)

class PartitionTextExplainer(PartitionExplainer, FeatureAttributions):
    '''
    Text classification-based partition explanation (using HuggingFace Transformers API). 

    Args:
      model (function): "black box" prediction function that takes an input array of shape (n samples, m features)
      and outputs an array of n predictions.
      tokenizer (string or callable): tokenizer used to break apart strings during masking of the text
      categories: list of label names for the given classification problem

    Attributes:
      explainer: shap PartitionExplainer object
      shap_values: the resulting shap value estimations on the target examples 

    Methods:
      visualize: display the text plot of the of the target example(s)

    Reference:
    https://shap-lrjball.readthedocs.io/en/latest/generated/shap.PartitionExplainer.html 
    '''
    def __init__(self, task_type, model, labels, tokenizer):
        FeatureAttributions.__init__(self)
        PartitionExplainer.__init__(self)
        self.shap_values = None
        self.explainer = self.shap.Explainer(model, masker=self.shap.maskers.Text(tokenizer=tokenizer), output_names=labels)

    def run_explainer(self, target_text, max_evals=64):
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
    '''
    Calculate the gradients of the output with respect to the inputs.

    Args:
      model (pytorch.nn.Module): a differentiable model to be interpreted

    Attributes:
      saliency: the captum saliency object
      grads: the gradients calculated from the saliency algorithm (created only after visualize() is called)

    Reference:
    https://captum.ai/docs/attribution_algorithms    
    '''
    def __init__(self, model):
        from captum.attr import Saliency

        self.saliency = Saliency(model)

    def visualize(self, input, labels, original_image, imageTitle):
        '''
        Visualize the saliency result with a blended heatmap

        Args:
          input (pytorch.Tensor): the image to be used for interpretation - requires_grad must be set to True.
          labels (pytorch.Tensor): list of the label names
          original_image (numpy.ndarray): the original image to be interpreted
          imageTitle (string): title of the visualization
        '''
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
    '''
    Approximate the integration of gradients with respect to inputs along the path from 
    a given input.

    Args:
      model (pytorch.nn.Module): a differentiable model to be interpreted

    Attributes:
      ig: the captum integrated gradients object
      attr_ig: the integrated gradients resulting from from ig.attribute() (created only after visualize() is called)
    Reference:
    https://captum.ai/docs/attribution_algorithms 
    '''
    def __init__(self, model):
        from captum.attr import IntegratedGradients

        self.ig = IntegratedGradients(model)

    def visualize(self, input, labels, original_image, imageTitle):
        '''
        Visualize the integrated gradients result with a blended heatmap

        Args:
          input (pytorch.Tensor): the image to be used for interpretation - requires_grad must be set to True.
          labels (pytorch.Tensor): list of the label names
          original_image (numpy.ndarray): the original image to be interpreted
          imageTitle (string): title of the visualization
        '''
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
    '''
    Approximate attributions using DeepLIFT's back-propagation based algorithm.

    Args:
      model (pytorch.nn.Module): a differentiable model to be interpreted

    Attributes:
      dl: the captum deepLIFT object
      attr_dl: captum's DeepLIFT attributions resulting from dl.attribute() (created only after visualize() is called)

    Reference:
    https://captum.ai/docs/attribution_algorithms 
    '''
    def __init__(self, model):
        from captum.attr import DeepLift

        self.dl = DeepLift(model)

    def visualize(self, input, labels, original_image, imageTitle):
        '''
        Visualize the DeepLIFT attributions with a blended heatmap

        Args:
          input (pytorch.Tensor): the image to be used for interpretation - requires_grad must be set to True.
          labels (pytorch.Tensor): list of the label names
          original_image (numpy.ndarray): the original image to be interpreted
          imageTitle (string): title of the visualization
        '''
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
    '''
    Use the Gaussian kernel to average the integrated gradients attributions via noise tunneling.

    Args:
      model (pytorch.nn.Module): a differentiable model to be interpreted

    Attributes:
      ig: the captum integrated gradients object
      nt: the captum noise tunnel object
      attr_ig_nt: resulting attributions from nt.attribute() (created only after visualize() is called)

    Reference:
    https://captum.ai/docs/attribution_algorithms 
    '''
    def __init__(self, model):
        from captum.attr import IntegratedGradients
        from captum.attr import NoiseTunnel

        self.ig = IntegratedGradients(model)
        self.nt = NoiseTunnel(self.ig)

    def visualize(self, input, labels, original_image, imageTitle):
        '''
        Visualize the smooth grad attributions with a blended heatmap

        Args:
          input (pytorch.Tensor): the image to be used for interpretation - requires_grad must be set to True.
          labels (pytorch.Tensor): list of the label names
          original_image (numpy.ndarray): the original image to be interpreted
          imageTitle (string): title of the visualization
        '''
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
    '''
    Approximate attributions using perturbation by replacing each input feature with a reference
    value and computing the difference 

    Args:
      model (pytorch.nn.Module): a differentiable model to be interpreted

    Attributes:
      ablator: the captum feature ablation object
      fa_attr: attributes resulting from ablator.attribute() (created only after visualize() is called)
    
    Reference:
    https://captum.ai/docs/attribution_algorithms 
    '''
    def __init__(self, model):
        from captum.attr import FeatureAblation

        self.ablator = FeatureAblation(model)

    def visualize(self, input, labels, original_image, imageTitle):
        '''
        Visualize the feature ablation attributions with a blended heatmap

        Args:
          input (pytorch.Tensor): the image to be used for interpretation - requires_grad must be set to True.
          labels (pytorch.Tensor): list of the label names
          original_image (numpy.ndarray): the original image to be interpreted
          imageTitle (string): title of the visualization
        '''
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
    """
    return PipelineExplainer('sentiment-analysis', model)(data)
