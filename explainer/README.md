# Intel® Explainable AI Tools

## Features
* Allows injection of XAI methods into python workflows/notebooks without requiring version compatibility of resident packages in the active python environment.
* Enables a seamless way to include the best or most relevant XAI methods from any external package/toolkit to explain models/data from within the pluggable environment without modifying the resident environment.
* Is extensible, leveraging python’s entry_points specification.
* Is compliant with python’s importlib and contextlib APIs, both part of standard python 3.
* Provides an avenue for community contributions. Plugin architectures have had tremendous success in python’s ecosystem. New plugins are created as wheels and published to python’s pypi or used locally on disk.

## Build and Install
Requirements:
* Linux system (or WSL2 on Windows)
* git
* required python version: >=3.9,<3.10
* `apt-get install build-essential python3-dev`

1. Clone this repo and navigate to the repo directory:
   ```
   git clone https://github.com/intel-innersource/frameworks.ai.explainable-ai.git

   cd frameworks.ai.explainable-ai/explainer/
   ```
2. Create and activate a Python3 virtual environment using `virtualenv`:
   ```
   python3 -m virtualenv xai_env
   source xai_env/bin/activate
   ```

   Or `conda`:
   ```
   conda create --name xai_env python=3.9
   conda activate xai_env
   ```
3. Install this tool with `make explainer`.
4. Optional: Build our documentation
   ```
   make docs-install

   make docs

   make docs-serve
   ```
   This serves the documentation locally on 127.0.0.1:9010
   You can also set `LISTEN_IP` and `LISTEN_PORT` values to modify the default behaviour.
   ```
   LISTEN_PORT=9999 make docs-serve
   ```

## Getting Started with the CLI

Use `explainer --help` to see the list of CLI commands. More detailed information on each
command can be found using `explainer <command> --help` (like `explainer export --help`).

| Commands | Description | 
|----------|-----------|
|build | builds a wheel and moves it to explainer/plugins|
|create | creates a yaml file under explainer/explainers|
|extract | unpacks a wheel file to explainer/plugins|
|generate | generates a template plugin directory under explainer/plugins|
|import | imports an explainable's functionality as defined by ExplainerSpec|
|info | shows info about an explainer under explainer.explainers|
|install | installs the plugin (wheel) under the directory explainer/explainers|
|list | lists available explainers|
|uninstall |uninstalls the plugin directory under the directory explainer/explainers|
|update | calls the generate, build and install apis|

## Running Notebooks

Each of our notebooks require entry points.

Run the entry point install command for the desired notebook. You many need to restart your environment for the Entry Point to be available within your environment. 

| Notebook | Entry Point Install Command | 
|----------|-----------|
|[Explaining Deep Learning Models](/docs/explainer/examples/ExplainingDeepLearningModels.ipynb)| `explainer install feature_attributions_explainer` & `explainer install metrics_explainer`|
|[Explaining Heart Disease](/docs/explainer/examples/heart_disease.ipynb)| `explainer install feature_attributions_explainer` & `explainer install metrics_explainer`|
|[Explaining Language Model Activations](/docs/explainer/examples/model_layers.ipynb)| `explainer install lm_layers_explainer`|
|[Explaining Text Classification](/docs/explainer/examples/partitionexplainer.ipynb)|  `explainer install feature_attributions_explainer` & `explainer install metrics_explainer`|
|[TorchVision CIFAR Interpret](/docs/explainer/examples/TorchVision_CIFAR_Interpret.ipynb)| `explainer install feature_attributions_explainer` & `explainer install metrics_explainer`|
|[Explaining Vision Transformers (Vit) ](/docs/explainer/examples/vit_transformer.ipynb)| `explainer install feature_attributions_explainer` & `explainer install metrics_explainer`|
