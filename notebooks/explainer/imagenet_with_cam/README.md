# Explaining RESNET50 ImageNet Classification Using the CAM Explainer 

This notebook demonstrates how to use the CAM explainer API to explain an ImageNet classification example using a RESNET50 CNN from the Torch Vision model hub and also from the TF keras.applications model hub. 

`ExplainingImageClassification.ipynb` performs the following steps:
1. Install and import dependencies
2. Load the ImageNet example image to be analyzed
3. Load the ResNet50 CNN trained on ImageNet
4. Visualize the ResNet50 classification of the loaded image using `xgradcam()`
5. Redo steps above for TF keras.applications ResNet50 model and using `tf_gradcam()`

## Running the notebook

To run `ExplainingImageClassification.ipynb`, install the following dependencies:
1. [Intel® Explainable AI](https://github.com/IntelAI/intel-xai-tools)


## References

pytorch-grad-cam GitHub Project - https://github.com/jacobgil/pytorch-grad-cam
TensorFlow implementation - https://github.com/ismailuddin/gradcam-tensorflow-2/blob/master/notebooks/GradCam.ipynb