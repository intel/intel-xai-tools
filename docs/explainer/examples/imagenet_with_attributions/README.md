# Explaining VGG19 ImageNet Classification Using the Attributions Explainer

This notebook demonstrates how to use the attributions explainer API to explain an ImageNet classification example using a VGG19 CNN from the TensorFlow model hub. 

`ExplainingDeepLearningModels.ipynb` performs the following steps:
1. Import libraries
2. Load the ImageNet example image to be analyzed
3. Load the VGG19 CNN trained on ImageNet
4. Visualize the VGG19 classification of the loaded image using `gradient_explainer()`
5. Create a custom CNN and visualize with `deep_explainer()` _(INCOMPLETE)_

## Running the notebook

To run `ExplainingDeepLearningModels.ipynb`, install the following dependencies:
1. [Intel® Explainable AI](https://github.com/intel-innersource/frameworks.ai.explainable-ai)

## References

1. SHAP GitHub Project - https://github.com/slundberg/shap
2. SHAP Documentations - https://shap.readthedocs.io/en/latest/index.html
3. SHAP Image Explainers - https://github.com/slundberg/shap/tree/master/notebooks/image_examples/image_classification
4. DeepExplainer Reference - "Deep Learning Model Interpretation Using SHAP" - https://towardsdatascience.com/deep-learning-model-interpretation-using-shap-a21786e91d16
4. Some of the utility functions and code are taken from the GitHub Repository of the author - Aditya Bhattacharya https://github.com/adib0073