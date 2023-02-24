# Explaining Custom CNN CIFAR-10 Classification Using the Attributions Explainer 
This notebook demonstrates how to use the attributions explainer API to explain the CIFAR-10 dataset image classification example using a Custom PyTorch CNN. 

`TorchVision_CIFAR_Interpret.ipynb` performs the following steps:
1. Import dependencies
2. Load the CIFAR-10 dataset from TorchVision hub
3. Design the PyTorch CNN model
4. Train the CNN
5. Visualize the custom CNN classifications using `saliency()`, `integratedgradients()`, `deeplift()`, `smoothgrad()` and `featureablation()`

## Running the notebook

To run `TorchVision_CIFAR_Interpret.ipynb`, install the following dependencies:
1. [Intel® Explainable AI](https://github.com/intel-innersource/frameworks.ai.explainable-ai)

## References
