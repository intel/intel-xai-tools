# Explaining Custom CNN MNIST Classification Using the Attributions Explainer

This notebook demonstrates how to use the attributions explainer API to explain an MNIST classification example using a Custom CNN. It also includes the metrics explainer API for error analysis.

`mnist.ipynb` performs the following steps:
1. Import libraries
2. Design the PyTorch custom CNN
3. Train the CNN on the MNIST dataset
4. Predict on the MNIST test data
5. Visualize test performance with `confusion_matrix()` and `plot()` functions
6. Visualize the CNN test classifications using `gradient_explainer()` and `deep_explainer()`

## Running the notebook

To run `mnist.ipynb`, install the following dependencies:
1. [Intel® Explainable AI](https://github.com/IntelAI/intel-xai-tools)
2. pip install jupyter-dash

## References

1. SHAP GitHub Project - https://github.com/slundberg/shap
2. SHAP MNIST example - "Pytorch Deep Explainer MNIST example": https://github.com/slundberg/shap/blob/master/notebooks/image_examples/image_classification/PyTorch%20Deep%20Explainer%20MNIST%20example.ipynb