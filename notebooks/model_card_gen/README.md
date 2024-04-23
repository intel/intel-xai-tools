# Model Card Generator Tutorial Notebooks
This directory has Jupyter notebooks that demonstrate model card generation using Intel® Explainable AI Tools.

| Notebook | Domain: Use Case | Framework| Description |
| ---------| ---------|----------|-------------| 
| [Generating a Model Card with PyTorch](model_card_generation_with_pytorch) | Numerical/Categorical: Tabular Classification | PyTorch | Demonstrates training a multilayer network using the "Adult" dataset from the UCI repository to predict whether a person has a salary greater or less than $50,000. The Model Card Generator is then used to create a model card with interactive graphics to analyze the model. |
| [Detecting Issues in Fairness by generating a Model Card from TensorFlow Estimators](compas_with_model_card_gen) | Numerical/Categorical: Tabular Classification  | TensorFlow | Utilizes a TFX pipeline to train and evaluate a model using the COMPAS (Correctional Offender Management Profiling for Alternative Sanctions) dataset to generate a risk score indended to determine a defendant's likelihood of reoffending. The Model Card Generator is then used to create interative graphics visualizing racial bias in the model's predictions. |
| [Creating Model Card for Toxic Comments Classification in TensorFlow](toxic_comments_classification) | Numerical/Categorical: Tabular Classification | TensorFlow | Adapts a [TensorFlow Fairness Exercise notebook](https://colab.research.google.com/github/google/eng-edu/blob/main/ml/pc/exercises/fairness_text_toxicity_part1.ipynb?utm_source=practicum-fairness&utm_campaign=colab-external&utm_medium=referral&utm_content=fairnessexercise1-colab#scrollTo=2z_xzJ40j9Q-) to use the Model Card Generator. The notebook trains a model to detect toxicity in online coversations and graphically analyzes accuracy metrics by gender. |

*Other names and brands may be claimed as the property of others. [Trademarks](http://www.intel.com/content/www/us/en/legal/trademarks.html)
