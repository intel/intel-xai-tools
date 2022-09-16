(explainer)=
# Explainer

The explainer tool provides a means to quickly integrate XAI methods into existing python environments such as workflows or notebooks.
XAI methods are grouped as explainer plugins where each plugin provides various XAI techniques from one or more Toolkits.

## Features:

- composable: allowing easy injection of XAI methods into existing python workflows|notebooks in 2-3 lines of code.
- extensible: plugin based architecture that includes many popular toolkits
- community focused: community contributions can be added to pypi as wheels

## Current Plugins:

| **Plugin**              | **Description**      | **Focus**                       | **Types of Models**  |
|-------------------------|----------------------|---------------------------------|----------------------|
| lm_layer_explainer      | nlp transformers     | Input Saliency                  | BERT, GPT-2          |
| lm_classifier_explainer | nlp classification   | Layerwise Relevance Propagation | BERT, GPT-2          |
| lm_features_explainer   | feature attributions | Feature Permutations            | VGG                  |



## Example Syntax:

```
from explainer.explainers import lm_layers_explainer

lm_layers_explainer['layer_activations'](model, text)
```

## Innovations:

- uses python's import mechanism to read a yaml file that is then used to load a python environment. This environment is ephemeral or pluggable and inserted at the head of sys.path.
- the ModuleSpec returned from import has been extended as a subclass to hold:
  - a dictionary of entry_points grouped by the import name that are listed in the pluggable environment's dist-info (entry_points.txt)
  - an optional context manager that will unload the environment when used with python's with statement
    - the context manager is similar to python's contextmanager for files but used for pluggable environments.
  - model as a python Resource <- an attribute in the yaml file.
  - dataset as a python Resource <- an attribute in the yaml file.
