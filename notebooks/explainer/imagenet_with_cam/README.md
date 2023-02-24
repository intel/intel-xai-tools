# Explaining RESNET50 ImageNet Classification Using the CAM Explainer 

This notebook demonstrates how to use the CAM explainer API to explain an ImageNet classification example using a RESNET50 CNN from the Torch Vision model hub. 

`ExplainingImageClassification.ipynb` performs the following steps:
1. Install and import dependencies
2. Load the ImageNet example image to be analyzed
3. Load the ResNet50 CNN trained on ImageNet
4. Visualize the ResNet50 classification of the loaded image using `xgradcam()`

## Running the notebook

To run `ExplainingImageClassification.ipynb`, install the following dependencies:
1. [Intel® Explainable AI](https://github.com/intel-innersource/frameworks.ai.explainable-ai)
2. `pip install scikit-image`

## References

pytorch-grad-cam GitHub Project - https://github.com/jacobgil/pytorch-grad-cam