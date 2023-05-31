# CAM (Class Activation Mapping)

```{code-cell} python3
from explainer import cam
```

CAM is an approach which localizes regions in the image responsible for a class prediction. 
Here, for object classification model, we support visualization of XGradCAM, which is the state-of-the art CAM method by
utilizing [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) package. 
We also support visualization of EigenCAM which is a fast method for object detection models. 

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
