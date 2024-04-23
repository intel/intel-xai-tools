# CAM (Class Activation Mapping)

| Method   | Decription                                                                                                 |
|------------|------------------------------------------------------------------------------------------------------------|
| tf_gradcam | Explain predictions with gradient-based class activation maps with the  TensorFlow|


```python3
from intel_ai_safety.explainer.cam import tf_cam

target_class = 281
target_layer = tf_resnet50.get_layer('conv5_block3_out')
gcam = tf_cam.tf_gradcam(tf_resnet50, target_layer, target_class, dog_cat_image)

```

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
