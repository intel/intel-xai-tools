---
file_format: mystnb
kernelspec:
  name: xai-oneapi-aikit
---
# Feature Attributions

```{code-cell} xai-oneapi-aikit
:tags: [remove-input]
from explainer.explainers import feature_attributions_explainer
```

This plugin provides a set of functions to explore visualize and understand their model's features. 
This plugin currently utilizes SHAP (SHapley Additive exPlanations) which is an approach to 
explain the output of any machine learning model. Feature Attributions are an approach to 
explaining a model's predictions based on how the model has weighted features it's been trained on.

## Algorithm

Determining feature importance within a model is done by utilizing the SHAP package.
Shap accepts black box models and provides explanations through a game theoretic approach.

## Environment
- Jupyter Notebooks

## XAI Methods
- Shap Explainers
  - Deep Explainer - provides explanations on Deep Learning models
  - Gradient Explainer
  - Kernel Explainer
  - Partition Explainer
  - Zero Shot Explainer

- To Be Added:
- Captum 
  - Integrated Gradients
  - Deep Lift  

## Toolkits
- Shap

- To Be Added:
- Captum

## References
[SHAP](https://github.com/slundberg/shap)
[Captum](https://github.com/pytorch/captum)


```{eval-rst}

.. automodule:: feature_attributions_explainer
   :noindex:
   :members:

```
