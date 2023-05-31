# CAM (Class Activation Mapping)

```python3
from explainer import cam
```

CAM is an approach which localizes regions in the image responsible for a class prediction. 
Here, for object classification model, we support visualization of XGradCAM, which is the state-of-the art CAM method by
utilizing [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) package. 
We also support visualization of the base gradCAM method and EigenCAM which is a fast method for object detection models. 

## Algorithm
CAM takes the input image and the trained model as input, and generates visualization by highlighting
the region in the corresponding image that are responsible for the classification decision.

## Environment
- Jupyter Notebooks

## XAI Methods
- GradCAM
- XGradCAM
- EigenCAM
## Toolkits
- pytorch-grad-cam

## References
[pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)<br> 
[TF GradCAM Implementation](https://github.com/ismailuddin/gradcam-tensorflow-2/blob/master/notebooks/GradCam.ipynb)<br>
[Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391) 
