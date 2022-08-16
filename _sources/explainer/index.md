(explainer)=
# Explainer

## Features:

- allows injection of xai methods into python workflows|notebooks without requiring version compatibility of resident packages in the active python environment.
- enables a seamless way to include the best or most relevant xai methods from any external package|toolkit to explain models|data from within the pluggable environment without modifying the resident environment.
- is extensible, leveraging python's entry_points specification.
- is compliant with python's importlib and contextlib APIs, both part of standard python 3.
- provides an avenuse for community contributions. Plugin architectures have had tremendous success in python's ecosystem. New plugins are created as wheels and published to python's pypi or used locally on disk.

## Syntax:

```
from explainer.explainers import model_layers

with model_layers as m:
  m.layer_attributions(model, text)
  m.layer_predictions(model, text)
```


## Innovations:

- uses python's import mechanism to read a yaml file that is then used to load a python environment. This environment is ephemeral or pluggable and inserted at the head of sys.path.
- the ModuleSpec returned from import has been extended as a subclass to hold:
  - a dictionary of entry_points grouped by the import name that are listed in the pluggable environment's dist-info (entry_points.txt)
  - an optional context manager that will unload the environment when used with python's with statement
    - the context manager is similar to python's contextmanager for files but used for pluggable environments.
  - model as a python Resource <- an attribute in the yaml file.
  - dataset as a python Resource <- an attribute in the yaml file.


## Plugin Naming Conventions (for various types of explainers):

- model-<plugin>: eg model-layers, model-posthoc, model-attributions
- data-<plugin>: eg data-perf, data-feature-permutation
