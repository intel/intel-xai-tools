---
file_format: mystnb
kernelspec:
  name: python3
---
# Feature Attributions

```{code-cell} python3
:tags: [remove-input]
from explainer.explainers import feature_attributions_explainer
```

```{mermaid}
graph LR
A(feature_attributions_explainer) --> B(deep_explainer)
A --> C(deeplift)
A --> D(featureablation)
A --> E(gradient_explainer)
A --> F(integratedgradients)
A --> G(kernel_explainer)
A --> H(partition_explainer)
A --> I(saliency)
A --> J(smoothgrad)
click B "/explainer/feature_attributions_explainer.html#feature_attributions_explainer.deep_explainer" "deep_explainer"
click C "/explainer/feature_attributions_explainer.html#feature_attributions_explainer.deeplift" "deeplift"
click D "/explainer/feature_attributions_explainer.html#feature_attributions_explainer.featureablation" "featureablation"
click E "/explainer/feature_attributions_explainer.html#feature_attributions_explainer.gradient_explainer" "gradient_explainer"
click F "/explainer/feature_attributions_explainer.html#feature_attributions_explainer.integratedgradients" "integratedgradients"
click G "/explainer/feature_attributions_explainer.html#feature_attributions_explainer.kernel_explainer" "kernel_explainer"
click H "/explainer/feature_attributions_explainer.html#feature_attributions_explainer.partition_explainer" "partition_explainer"
click I "/explainer/feature_attributions_explainer.html#feature_attributions_explainer.saliency" "saliency"
click J "/explainer/feature_attributions_explainer.html#feature_attributions_explainer.smoothgrad" "smoothgrad"
```

Feature Attributions are an approach to explaining a model's predictions based on how the model has weighted features it's been trained on.
This set of functions visualizes features by utilizing [SHAP](https://github.com/slundberg/shap) (SHapley Additive exPlanations): an approach to 
explain the output of ML/DL models. 

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
- Captum 
  - Integrated Gradients
  - Deep Lift  
  - Saliency
  - Smooth Grad via Noise Tunnel
  - Feature Ablation

## Toolkits
- Shap
- Captum

## References
[SHAP](https://github.com/slundberg/shap)\
[Captum](https://github.com/pytorch/captum)


## Entry Points

```{eval-rst}

.. automodule:: feature_attributions_explainer
   :members:

```

