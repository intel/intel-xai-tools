### [Attributions](attributions)

| Method              | Decription                                                                                  |
|---------------------|---------------------------------------------------------------------------------------------|
| explainer           | Return native Shap references as attributes                                                 |
| kernel_explainer    | Explain predictions using SHAP's kernel method                                              |
| deep_explainer      | Explain predictions using SHAP's deep method                                                |
| gradient_explainer  | Explain predictions using SHAP's gradient method                                            |
| partition_explainer | Explain predictions using SHAP's partition method                                           |
| saliency            | Explain predictions using Captum's saliency method                                          |
| integratedgradients | Explain predictions using Captum's integrated gradients method                              |
| deeplift            | Explain predictions using Captum's deep lift method                                         |
| smoothgrad          | Explain predictions using Captum's noise tunnel smooth gradient method                      |
| featureablation     | Explain predictions using Captum's feature ablation method                                  |
| zero_shot           | ...                                                                                         |
| sentiment_analyis   | Explain HuggingFace pipeline predictions the SHAP explainer methods                         |

### [CAM](cam)

| Method   | Decription                                                                                                 |
|------------|------------------------------------------------------------------------------------------------------------|
| xgradcam   | Explain predictions with axiom-based gradient-based class activation maps using pytorch-grad-cam methods   |
| eigancam   | Explain predictions with eigan-based class activation maps using pytorch-grad-cam methods                  |
| tf_gradcam | Explain predictions with gradient-based class activation maps with the  TensorFlow|

### [Metrics](metrics)

| Method | Decription                                                                                  |
|--------|---------------------------------------------------------------------------------------------|
| confusion_matrix | Visualize classifier performance  via  a contingency table visualization          |
| plot   | Visualize classifier performance via ROC/PR values over a spread of probability threasholds |
| pstat  | Report the execution summary of a given snippet of code using the cProfile run method       |
