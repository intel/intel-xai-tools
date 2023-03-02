# XAI Notebooks

This directory has Jupyter notebooks that demonstrate explainable AI with the Intel® Explainable AI Tool. 

## Explainer Tutorial Notebooks

| Notebook | Domain: Use Case | Framework| Description |
| ---------| ---------|----------|-------------|
| [Explaining ResNet50 ImageNet Classification Using the CAM Explainer](explainer/imagenet_with_cam) | CV: Image Classification | PyTorch and Intel® Explainable AI API | Demonstrates how to use the CAM explainer API to explain an ImageNet classification example using a ResNet50 CNN from the Torch Vision model hub. |
| [Explaining VGG19 ImageNet Classification Using the Attributions Explainer](explainer/imagenet_with_attributions) | CV: Image Classification | TensorFlow and Intel® Explainable AI API | Demonstrates how to use the attributions explainer API to explain an ImageNet classification example using a VGG19 CNN from the TensorFlow model hub. |
| [Explaining Custom CNN MNIST Classification Using the Attributions Explainer](explainer/imagenet_with_attributions) | CV: Image Classification | PyTorch and Intel® Explainable AI API | Demonstrates how to use the attributions explainer API to explain an MNIST classification example using a Custom PyTorch CNN. |
| [Explaining Custom NN NewsGroups Classification Using the Attributions Explainer](explainer/newsgroups_with_attributions_and_metrics) | NLP: Text Classification | PyTorch and Intel® Explainable AI API | Demonstrates how to use the attributions explainer API to explain a NewsGroups dataset text classification example using a Custom TensorFlow NN. |
| [Explaining Custom CNN CIFAR-10 Classification Using the Attributions Explainer](explainer/cifar_with_attributions) | CV: Image Classification | PyTorch and Intel® Explainable AI API | Demonstrates how to use the attributions explainer API to explain the CIFAR-10 dataset image classification example using a Custom PyTorch CNN. |
| [Multimodal Breast Cancer Detection Explainability using the Intel® Explainable AI  API](explainer/multimodal_cancer_detection) | CV: Image Classification & NLP: Text Classification| PyTorch, HuggingFace, Intel® Explainable AI API & Intel® Transfer Learning Tool API | Demonstrates how to use the attributions and metrics explainer API's to explain the classification of a text and image breast cancer dataset using a PyTorch ResNet50 CNN and a HuggingFace ClinicalBert Transformer. |
| [Explaining Fine Tuned Text Classifier with PyTorch using the Intel® Explainable AI  API](explainer/transfer_learning_text_classification) | NLP: Text Classification| PyTorch, HuggingFace, Intel® Explainable AI API & Intel® Transfer Learning Tool API | Demonstrates how to use the attributions explainer API's to explain the classification of a text using  HuggingFace Transformer. |