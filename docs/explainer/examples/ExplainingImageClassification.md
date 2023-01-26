---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Explaining Image Classification Models

### Objective
The goal of this notebook is to explore various CAM methods for image classification models. 
For now, we only support XGradCAM method, which is the state-of-the-art CAM method. 
```{code-cell} ipython3
import warnings
warnings.filterwarnings('ignore')
```

### Installing the module
```{code-cell} ipython3
from explainer.explainers import cam_explainer
```
### Loading the modules
```{code-cell} ipython3
import torch
from torchvision.models import resnet50, ResNet50_Weights
from skimage import io
import matplotlib.pyplot as plt
```

## Using XGradCAM

### Loading the input image
Load the input image as a numpy array in RGB order. 
```{code-cell} ipython3
image = io.imread("https://raw.githubusercontent.com/jacobgil/pytorch-grad-cam/master/examples/both.png")

plt.imshow(image)
plt.show()
```

### Loading the model
Load the trained model depending on how the model was saved. 
If you have your trained model, load it from the model's path using 'torch.load()'.
```{code-cell} ipython3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2) # Let's use ResNet50 trained on ImageNet as our model
```
We need to choose the target layer (normally the last convolutional layer) to compute CAM for. 
Simply printing the model will give you some idea about the name of layers and their specifications.
Here are some common choices:
- FasterRCNN: model.backbone
- Resnet18 and 50: model.layer4
- VGG and densenet161: model.features

```{code-cell} ipython3
targetLayer = "model.layer4"
```
We need to specify the target class as an integer to compute CAM for. 
This can be specified with the class index in the range [0, NUM_OF_CLASSES-1] based on the training dataset. 
For example, the index of the class 'tabby cat' is 281 in ImageNet. 
If targetClass is None, the highest scoring category will be used. 
 
```{code-cell} ipython3
targetClass = 281
```

### Visualization
```{code-cell} ipython3
xgradcam = cam_explainer.xgradcam(model, targetLayer, targetClass, image, device)
xgradcam.visualize()
```

## References
pytorch-grad-cam GitHub Project - https://github.com/jacobgil/pytorch-grad-cam
