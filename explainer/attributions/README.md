# Feature Attributions

```python3
from explainer import attributions
```

Feature Attributions are an approach to explaining a model's predictions based on how the model has weighted the features it's been trained on.
This set of functions visualizes features by utilizing [SHAP](https://github.com/slundberg/shap) (SHapley Additive exPlanations): an approach to explain the output of ML/DL models. 

## Algorithm

Determining feature importance within a model is done by utilizing the SHAP package.
Shap accepts black box models and provides explanations through a game theoretic approach.

## Environment
- Jupyter Notebooks

## XAI Methods
- Shap Explainers
  - Deep Explainer
  - Gradient Explainer
  - Kernel Explainer
  - Partition Explainer
  - Zero Shot Explainer
  - Sentiment Analysis
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
