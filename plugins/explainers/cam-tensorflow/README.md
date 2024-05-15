# CAM (Class Activation Mapping)

| Method   | Decription                                                                                                 |
|------------|------------------------------------------------------------------------------------------------------------|
| tf_gradcam | Explain predictions with gradient-based class activation maps with the  TensorFlow|


CAM is an approach which localizes regions in the image responsible for a class prediction. 
Here, for object classification model, we support visualization of GradCAM, which is the state-of-the art CAM method. 

## Algorithm
CAM takes the input image and the trained model as input, and generates visualization by highlighting
the region in the corresponding image that are responsible for the classification decision.

## Environment
- Jupyter Notebooks


## References
[TF GradCAM Implementation](https://github.com/ismailuddin/gradcam-tensorflow-2/blob/master/notebooks/GradCam.ipynb)<br>
[Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391) 
