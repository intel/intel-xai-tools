# Metrics

## Metrics

| Method | Decription                                                                                  |
|--------|---------------------------------------------------------------------------------------------|
| confusion_matrix | Visualize classifier performance  via  a contingency table visualization          |
| plot   | Visualize classifier performance via ROC/PR values over a spread of probability threasholds |
| pstat  | Report the execution summary of a given snippet of code using the cProfile run method       |


```python
from intel_ai_safety.explainer import metrics
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
