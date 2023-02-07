---
file_format: mystnb 
kernelspec:
  name: python3
  orphan: true
---
# Metrics
 
```{code-cell} python3
:tags: [remove-input]
from explainer import metrics
```

```{mermaid}
graph LR
A(metrics_explainer) --> B(confusion_matrix)
A --> C(plot)
A --> D(pstats)
click B "/explainer/metrics_explainer.html#metrics_explainer.confusion_matrix" "confusion_matrix"
click C "/explainer/metrics_explainer.html#metrics_explainer.plot" "plot"
click D "/explainer/metrics_explainer.html#metrics_explainer.pstats" "pstats"
```

Several base metrics are provided for ML/DL classification models. These metrics cover model execution and performance and orient the data scientist to where there is potential for classification bias. 

## Algorithms
Provided with a classfication model's predictions and their corresponding ground truths, staple performance metrics can be calculated to determine prediction behaviors in the real world. These functions leverage scikit-learn and plotly (eventually) to calculate and visualize said metrics, respectively.

## Environment
- Jupyter Notebooks

## Metrics
- Performance metrics
  - Confusion Matrix
  - Performance Plots
- Execution metrics
  - Python profiler

## Toolkits
- Scikit-learn
- Plotly
- Python Profilers

## References

[Scikit-learn](https://github.com/scikit-learn/scikit-learn)\
[Plotly](https://github.com/plotly)\
[Python Profiler](https://github.com/python/cpython/blob/main/Lib/cProfile.py)

## Entry Points

```{eval-rst}

.. automodule:: metrics_explainer
   :members:

```
