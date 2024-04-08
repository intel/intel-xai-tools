# Explainer Tutorial Notebooks
This directory has Jupyter notebooks that demonstrate explainability with the Intel® Explainable AI Tools.

| Notebook | Domain: Use Case | Framework| Description |
| ---------| ---------|----------|-------------|
| [Explaining Image Classification and Object Detection Using the CAM Explainer](imagenet_with_cam) | CV: Image Classification & Object Detection | PyTorch*, TensorFlow* and Intel® Explainable AI API | Two separate notebooks that demonstrate how to use the CAM explainer API to explain ImageNet classification and detection examples using a ResNet50 CNN from the TorchVision & Torch model hub and TF's keras.applications model hub. |
| [Explaining Custom CNN MNIST Classification Using the Attributions Explainer](mnist_with_attributions_and_metrics) | CV: Image Classification | PyTorch and Intel® Explainable AI API | Demonstrates how to use the attributions explainer API to explain an MNIST classification example using a Custom PyTorch CNN. |
| [Explaining Custom NN NewsGroups Classification Using the Attributions Explainer](newsgroups_with_attributions_and_metrics) | NLP: Text Classification | PyTorch and Intel® Explainable AI API | Demonstrates how to use the attributions explainer API to explain a NewsGroups dataset text classification example using a Custom TensorFlow NN. |
| [Explaining Custom CNN CIFAR-10 Classification Using the Attributions Explainer](cifar_with_attributions) | CV: Image Classification | PyTorch and Intel® Explainable AI API | Demonstrates how to use the attributions explainer API to explain the CIFAR-10 dataset image classification example using a Custom PyTorch CNN. |
| [Multimodal Breast Cancer Detection Explainability using the Intel® Explainable AI API](multimodal_cancer_detection) | CV: Image Classification & NLP: Text Classification| PyTorch, HuggingFace, Intel® Explainable AI API & Intel® Transfer Learning Tool API | Demonstrates how to use the attributions and metrics explainer API's to explain the classification of a text and image breast cancer dataset using a PyTorch ResNet50 CNN and a HuggingFace ClinicalBert Transformer. |
| [Explaining Fine Tuned Text Classifier with PyTorch using the Intel® Explainable AI API](transfer_learning_text_classification) | NLP: Text Classification| PyTorch, HuggingFace, Intel® Explainable AI API & Intel® Transfer Learning Tool API | Demonstrates how to use the attributions explainer API's to explain the classification of a text using  HuggingFace Transformer. |
| [Explaining a Custom Neural Network Heart Disease Classification Using the Attributions Explainer ](heart_disease_with_attributions) | Numerical/Categorical: Tabular Classification | TensorFlow & Intel® Explainable AI API | Demonstrates how to use the attributions explainer API's to explain the classification of a Tabular data using a TensorFlow custom NN. |

## Running the Explainer Tutorial notebooks
Before running the notebooks, install the dependencies:
1. `pip install --no-cache-dir -r requirements.txt`
