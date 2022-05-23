# Use Cases

`````{tabbed} Definitions
````{panels}
:container: +full-width
:column: col-lg-4 px-2 py-2
:card: rounded
:header: bg-primary text-white text-center font-weight-bold
:body: bg-light text-left font-italic
```{dropdown} Business Owner
The business owner has a dataset (e.g. labelled images, videos, or product reviews) and a deployed model the data work with. They need to quickly need access using no code or very low code interface to get metrics and visualization informing the model’s behavior. They are a subject matter expert in their business domain and want to monitor whether their application has inherent risk around fairness, interpretability, privacy, and security. They need to quickly identify potential risk on a small data set to in their business area.
```
---
```{dropdown} Data Scientist
The Data Scientist understands the general data modeling workflow and knows their dataset very well. They may or may not be an expert in deep learning, but they are versed in the mathematics behind machine learning and are tasked with developing a model that balances high accuracy and good performance. They are responsible for selecting a model, training algorithm, and hyperparameters and they will be comfortable with programming languages and environments. They need to be able to experiment and iterate quickly and see detailed metrics reports.
```
---
```{dropdown} MLOps Engineer
The ML Engineer is responsible for deploying the selected model into production and updating the model if it is to be fine-tuned with new data on a regular basis. They will scale the training and/or production environment if required by the business. They might need to take a training workflow from a Data Scientist with most or all parameters defined and run it to convergence as efficiently as possible.
```
````
`````

````{tabbed} Use Cases
:selected:
* As a Data Scientist, i want to be able to see a set of features rated their relative importance for my model and dataset.
* As a Data Scientist, i want to be able to generate counterfactuals using XAI for my model and features.
* As a Data Scientist, i want to be able to detect and isolate bias in a model's prediction using XAI
* As a Data Scientist, I want to be able to better evaluate a confusion matrix using XAI
* As a Data Scientist, I want to be able to select a type of XAI from a choice of XAI methods that will use my model, dataset and features.
* As a ML Engineer, I want an easy way to create and deploy a workflow using a Data Scientist's notebook
* As a Business Owner I need to assess the risk around fairness, interpretability, privacy, and security for a model.
* As a Data Scientist, I need a way to (1) identify problematic errors my model is making before deployment; and (2) understand why predictions are being made through visualizations, so I can fix them.
* As a Data Scientist, when using tabular data, I need identify the effect each feature is attributing to a model’s prediction.
* As a Data Scientist, when using tabular data, I need identify which features impact a model’s prediction most in aggregate.
* As a Data Scientist, when using image data, I need identify the effect each pixel is attributing to a model’s prediction.
* As a Data Scientist, I need to explore how different features values for a datapoint effects a model’s decision.
* As an ML /MLOps Engineer, I need a way to (1) detect problematic errors my model is making in real-time after deployment and (2) understand and interpret predictions made by your machine learning models in production through visualizations.
* * As a MLOps engineer, i want to be able to run explainer so that it monitors data drift for a model in production.
````

