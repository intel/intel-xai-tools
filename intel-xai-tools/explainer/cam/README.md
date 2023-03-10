---
file_format: mystnb
kernelspec:
  name: python3
---
# CAM (Class Activation Mapping)

```{code-cell} python3
:tags: [remove-input]
from explainer import cam
```

```{mermaid}
graph LR
A(cam_explainer) --> B(xgradcam)
A(cam_explainer) --> C(eigencam)
click B "/explainer/cam_explainer.html#cam_explainer.xgradcam" "xgradcam"
click C "/explainer/cam_explainer.html#cam_explainer.eigencam" "eigencam"
```
CAM is an approach which localizes regions in the image responsible for a class prediction. 
Here, for object classification model, we support visualization of XGradCAM, which is the state-of-the art CAM method by
utilizing [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) package. 
We also support visualization of EigenCAM which is extremely fast method for object detection model. 

## Algorithm
CAM takes the input image and the trained model as input, and generates visualization by highlighting
the region in the corresponding image that are responsible for the classification decision.

## Environment
- Jupyter Notebooks

## XAI Methods
- XGradCAM
- EigenCAM

## Toolkits
- pytorch-grad-cam

## References
[pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)


## Entry Points

```{eval-rst}

.. automodule:: cam_explainer
   :members:

```

