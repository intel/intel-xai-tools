---
file_format: mystnb
kernelspec:
  name: python3
---
(explainer)=
# Explainer

The explainer tool provides a means to quickly integrate XAI methods into existing python environments such as workflows or notebooks.
Explainable methods are grouped by type where each type of explainer provides various interpretability techniques from one or more Toolkits.

## Features

````{grid} 3

```{grid-item-card}
:text-align: center
:class-header: sd-font-weight-bold
:class-body: sd-font-italic
{octicon}`workflow` Composable
^^^
add to workflows in 2-3 lines of code
```

```{grid-item-card}
:text-align: center
:class-header: sd-font-weight-bold
:class-body: sd-font-italic
{octicon}`stack` Extensible
^^^
easy to add new external toolkits
```

```{grid-item-card}
:text-align: center
:class-header: sd-font-weight-bold
:class-body: sd-font-italic
{octicon}`package-dependencies` Community
^^^
contributions as python wheels
```

````


## Explanations

```{mermaid}
graph LR
A(Explainer) --> B(Feature Attributions)
A --> C(Language Models)
A --> D(Metrics)
A --> E(Class Activation Mapping)
click B "/explainer/feature_attributions_explainer.html#feature-attributions-explainer" "Feature Attributions"
click C "/explainer/lm_layers_explainer.html#language-model-explainer" "Language Models"
click D "/explainer/metrics_explainer.html#metrics-explainer" "Metrics"
click E "/explainer/cam_explainer.html#cam-explainer" "Class Activation Mapping"
```

````{grid} 3

```{grid-item-card} Feature Attributions
:text-align: center
:link: /explainer/feature_attributions_explainer.html#feature-attributions-explainer
[SHAP](https://github.com/slundberg/shap) explainers
```

```{grid-item-card}  Language Models
:text-align: center
:link: /explainer/lm_layers_explainer.html#language-model-explainer
transformer visualizations
```

```{grid-item-card}  Metrics
:text-align: center
:link: /explainer/metrics_explainer.html#metrics-explainer
profiling, confusion matrix
```

```{grid-item-card}  Class Activation Mapping
:text-align: center
:link: /explainer/cam_explainer.html#metrics-explainer
CAM explainers
```

````
