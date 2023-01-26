---
file_format: mystnb
kernelspec:
  name: python3
---
# CAM (Class Activation Mapping)

```{code-cell} python3
:tags: [remove-input]
from explainer.explainers import cam_explainer
```

```{mermaid}
graph LR
A(cam_explainer) --> B(xgradcam)
click B "/explainer/cam_explainer.html#cam_explainer.xgradcam" "xgradcam"
```
CAM is an approach which localizes regions in the image responsible for a class prediction. 
Here, we support visualization of XGradCAM, which is the state-of-the art CAM method by
utilizing [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) package. 

## Algorithm
CAM takes the input image and the trained model as input, and generates visualization by highlighting
the region in the corresponding image that are responsible for the classification decision.

## Environment
- Jupyter Notebooks

## XAI Methods
- XGradCAM

## Toolkits
- pytorch-grad-cam

## References
[pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)


## Entry Points

```{eval-rst}

.. automodule:: cam_explainer
   :members:

```

