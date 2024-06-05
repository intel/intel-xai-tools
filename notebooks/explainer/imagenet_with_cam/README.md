# Explaining Image Classification and Object Detection with Grad CAM and Eigan CAM Algorithms
This notebook directory has two notebooks demonstrating how to use the Class Activation Mapping API for image classification (`ExplainingImageClassification.ipynb`) and object detection (`ExplainingObjectDetection.ipynb`) task types.

## Explaining RESNET50 ImageNet Classification Using XGradCAM and TFGradCAM

This notebook demonstrates how to use the CAM explainer API to explain an ImageNet classification example using a RESNET50 CNN from the Torch Vision model hub and also from the TF keras.applications model hub. 

`ExplainingImageClassification.ipynb` performs the following steps:
1. Install and import dependencies
2. Load the ImageNet example image to be analyzed
3. Load the ResNet50 CNN trained on ImageNet
4. Visualize the ResNet50 classification of the loaded image using `x_gradcam()`
5. Redo steps above for TF keras.applications ResNet50 model and using `tf_gradcam()`

### Running the notebook

To run `ExplainingImageClassification.ipynb`, install the following dependencies:
1. [Intel® Explainable AI](https://github.com/Intel/intel-xai-tools)


### References

pytorch-grad-cam GitHub Project - https://github.com/jacobgil/pytorch-grad-cam
TensorFlow implementation - https://github.com/ismailuddin/gradcam-tensorflow-2/blob/master/notebooks/GradCam.ipynb

 
## Explaining FasterRCNN and YOLO ImageNet Object Detection Using EigenCAM

This notebook demonstrates how to use the CAM explainer API to explain two ImageNet object detection examples using a FasterRCNN and a YOLO model from the TorchVision and Torch model hubs, respectively. 

`ExplainingObjectDetection.ipynb` performs the following steps:
1. Install and import dependencies
2. Load the ImageNet example image to be analyzed
3. Load the FasterRCNN trained on ImageNet
4. Define the helper functions
4. Visualize the FasterRCNN detection of the loaded image using `eigencam()`
5. Redo steps above for the YOLO model

### Running the notebook

To run `ExplainingObjectDetection.ipynb`, install the following dependencies:
1. [Intel® Explainable AI](https://github.com/Intel/intel-xai-tools)
2. `pip install scikit-image ultralytics`


### References
pytorch-grad-cam GitHub tutorial for object detection with FasterRCNN - https://jacobgil.github.io/pytorch-gradcam-book/Class%20Activation%20Maps%20for%20Object%20Detection%20With%20Faster%20RCNN.html 
pytorch-grad-cam GitHub tutorial for object detection with YOLO - https://jacobgil.github.io/pytorch-gradcam-book/EigenCAM%20for%20YOLO5.html
