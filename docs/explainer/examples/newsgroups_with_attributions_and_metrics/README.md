# Explaining Custom NN NewsGroups Classification Using the Attributions Explainer

This notebook demonstrates how to use the attributions explainer API to explain a NewsGroups dataset text classification example using a Custom TensorFlow NN.

`partitionexplainer.ipynb` performs the following steps:
1. Import libraries
2. Fetch NewsGroups dataset
3. Vectorize text data
4. Design neural network
5. Train NN
6. Predict on the test data
7. Visualize test performance with `confusion_matrix()` and `plot()` functions
9. Visualize the custom NN test classifications using `partition_explainer()` and shap's bar and waterfall plots

## Running the notebook

To run `partitionexplainer.ipynb`, install the following dependencies:
1. [Intel® Explainable AI](https://github.com/intel-innersource/frameworks.ai.explainable-ai)
2. pip install jupyter-dash

## References

1. Explain Text Classification Models Using SHAP Values - https://coderzcolumn.com/tutorials/artificial-intelligence/explain-text-classification-models-using-shap-values-keras