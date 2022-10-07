(quickstart)=
# Quick Start


## Build and Install
Requirements:
* Linux system (or WSL2 on Windows)
* git
* python >= 3.9
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

## Getting Started with the CLI

Use `explainer --help` to see the list of CLI commands. More detailed information on each
command can be found using `explainer <command> --help` (like `explainer export --help`).

| Commands | Description | 
|----------|-----------|
|build | builds a wheel and moves it to explainer/plugins|
|extract | unpacks a wheel file to explainer/plugins|
|generate | generates a template plugin directory under explainer/plugins|
|import | imports an explainable's functionality as defined by ExplainerSpec|
|info | shows info about an explainer under explainer.explainers|
|list | lists available explainers|
|visualize | Plots an explanation JSON definition|


## Running Notebooks

Each of our notebooks require entry points.

Run the entry point install command for the desired notebook. You many need to restart your environment for the Entry Point to be available within your environment. 

| Notebook | Entry Point Install Command | 
|----------|-----------|
|<a href="/explainer/examples/model_layers.html#explaining-a-language-model-s-layer-activations">Explaining a language model's layer activations</a>|`explainer install lm_layers_explainer`|
