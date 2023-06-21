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

class XGradCAM(GradCAM):
    '''
    Holds the calculations for the axiom-based gradient-weighted class activation mapping (XgradCAM) of a 
    given image and PyTorch CNN.

    Args:
      model (torch.nn.Module): the CNN used for classification 
      target_layer (torch.nn.modules.container.Sequential): the convolution layer that you want to analyze (usually the last) 
      dims (tuple of ints): dimension of image (h, w)
      device (torch.device): torch.device('cpu') or torch.device('gpu') for PyTorch optimizations

        
    Attributes:
      model: the CNN being used
      target_layer: the target convolution being used 
      target_class: the target class being used
      image: the image being used
      dims: the dimensions of the image being used
      device: device being used by PyTorch

    Methods:
      visualize: superimpose the gradCAM result on top of the original image

    Reference:
       https://github.com/jacobgil/pytorch-grad-cam
    '''
    def __init__(self, model, target_layer, dims, device='cpu'):
        # set any frozen layers to trainable
        # gradcam cannot be calculated without it
        for param in model.parameters():
            if not param.requires_grad:
                param.requires_grad = True

        self.model = model
        self.target_layer = [target_layer]
        self.dims = dims
        self.device = device

    def run_explainer(self, image, target_class):
        '''
        Execute the axiom-based gradient-based class activation mapping algorithm on the image.

        Args: 
            image (numpy.ndarray): image to be analyzed with a shape (h,w,c)
            target_class (int): the index of the target class

        Returns:
            None
        '''
        from pytorch_grad_cam import XGradCAM, GuidedBackpropReLUModel
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
        import cv2
        import numpy as np

        self.model.eval().to(self.device)

        image = cv2.resize(image, self.dims)
        # convert to rgb if image is grayscale
        converted = False
        if len(image.shape) == 2:
            converted = True 
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        rgb_img = np.float32(image) / 255
        input_tensor = preprocess_image(rgb_img,
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        input_tensor = input_tensor.to(self.device)
        
        if target_class is None:
            targets = None
        else:
            targets = [ClassifierOutputTarget(target_class)]

        cam = XGradCAM(self.model, self.target_layer)

        # convert back to grayscale if that is the initial dim
        if converted:
            input_tensor = input_tensor[:, 0:1, :, :]

        grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=False,
                            eigen_smooth=False)
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        self.cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        gb_model = GuidedBackpropReLUModel(model=self.model, use_cuda=False)
        gb = gb_model(input_tensor, target_category=None)
        cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        self.cam_gb = deprocess_image(cam_mask * gb)
        self.gb = deprocess_image(gb)

    def visualize(self):
        '''
        Plot the axiom gradCAM, guided back propagation and guided axiom gradCAM from left
        to right, all superimposed on the target image.  
        '''

        import matplotlib.pyplot as plt
        import cv2
        
        fig = plt.figure(figsize=(10, 7))
        rows = 1
        columns = 3

        fig.add_subplot(rows, columns, 1)
        plt.imshow(cv2.cvtColor(self.cam_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("XGradCAM")

        fig.add_subplot(rows, columns, 2)
        plt.imshow(cv2.cvtColor(self.gb, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("Guided backpropagation")

        fig.add_subplot(rows, columns, 3)
        plt.imshow(cv2.cvtColor(self.cam_gb, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("Guided XGradCAM")

        print("XGradCAM, Guided backpropagation, and Guided XGradCAM are generated. ")



class EigenCAM:

    '''
    Holds the calculations for the eigan-based gradient-weighted class activation mapping (EiganCAM) of a 
    given image and PyTorch CNN for object detection.

    Args:
      model (torch.nn.Module): the CNN used for classification 
      target_layer (torch.nn.modules.container.Sequential): the convolution layer that you want to analyze (usually the last) 
      boxes (list): list of coordinates where the object is detected
      classes (list): list of classes that are predicted from boxes
      colors (list): list of colors corresponding to the classes
      reshape (function): the reshape transformation function responsible for processing the output tensors. Can be None
        if not needed for particular model (such as YOLO)
      image (numpy.ndarray): image to be analyzed with a shape (h,w,c)
      device (torch.device): torch.device('cpu') or torch.device('gpu') for PyTorch optimizations

        
    Attributes:
      model: the CNN being used
      target_layer: the target convolution being used 
      boxes: the list of coordinates being used
      classes: the list of classes being used
      colors: the list of colors being used for the classes
      reshape: the transformation function being used to process model output
      image: the image being used
      device: device being used by PyTorch

    Methods:
      visualize: superimpose the EiganCAM  result on top of the original image

    Reference:
       https://github.com/jacobgil/pytorch-grad-cam 
    '''

    def __init__(self, model, targetLayer, boxes, classes, colors, reshape, image, device):
        self.model = model
        self.targetLayer = targetLayer
        self.boxes = boxes
        self.classes = classes
        self.colors = colors
        self.reshape = reshape
        self.image = image
        self.device = device

    def visualize(self):
        from pytorch_grad_cam import EigenCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image, scale_cam_image
        import torchvision
        import torch
        import cv2
        import numpy as np
        from PIL import Image
        from IPython.display import display

        self.model.eval().to(self.device)

        rgb_img = np.float32(self.image) / 255
        transform = torchvision.transforms.ToTensor()
        input_tensor = transform(rgb_img)
        input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(self.device)

        self.targetLayer = [self.targetLayer]

        if self.reshape is None:
            cam = EigenCAM(self.model, self.targetLayer)
        else:
            cam = EigenCAM(self.model, self.targetLayer,
                           reshape_transform=self.reshape)
        targets = []
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=False,
                            eigen_smooth=False)
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        for x1, y1, x2, y2 in self.boxes:
            renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
        renormalized_cam = scale_cam_image(renormalized_cam)
        eigencam_image_renormalized = show_cam_on_image(rgb_img, renormalized_cam, use_rgb=True)
        for i, box in enumerate(self.boxes):
            color = self.colors[i]
            cv2.rectangle(
                eigencam_image_renormalized,
                (box[0], box[1]),
                (box[2], box[3]),
                color, 2
            )
            cv2.putText(eigencam_image_renormalized, self.classes[i], (box[0], box[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                        lineType=cv2.LINE_AA)

        display(Image.fromarray(np.hstack((cam_image, eigencam_image_renormalized))))

        print("EigenCAM is generated. ")



def x_gradcam(model, target_layer, target_class, image, dims, device='cpu'):
    gc = XGradCAM(model, target_layer, dims, device)
    gc.run_explainer(image, target_class)
    return gc

def eigencam(model, targetLayer, boxes, classes, colors, reshape, image, device):
    return EigenCAM(model, targetLayer, boxes, classes, colors, reshape, image, device)