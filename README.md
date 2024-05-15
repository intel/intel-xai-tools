# Intel® Explainable AI Tools

This repository provides tools for data scientists and MLOps engineers that have requirements specific to AI model interpretability.

## Overview

The Intel Explainable AI Tools are designed to help users detect and mitigate against issues of fairness and interpretability, while running best on Intel hardware.
There are two Python* components in the repository:

* [Model Card Generator](model_card_gen)
  * Creates interactive HTML reports containing model performance and fairness metrics
* [Explainer](explainer)
  * Runs post-hoc model distillation and visualization methods to examine predictive behavior for both TensorFlow* and PyTorch* models via a simple Python API including the following modules:
    * [Attributions](plugins/explainers/attributions): Visualize negative and positive attributions of tabular features, pixels, and word tokens for predictions
    * [CAM (Class Activation Mapping)](plugins/explainers/cam-pytorch): Create heatmaps for CNN image classifications using gradient-weight class activation CAM mapping
    * [Metrics](plugins/explainers/metrics): Gain insight into models with the measurements and visualizations needed during the machine learning workflow

## Get Started

### Requirements
* Linux system or WSL2 on Windows (validated on Ubuntu* 20.04/22.04 LTS)
* Python 3.9, 3.10
* Install required OS packages with `apt-get install build-essential python3-dev`
* git (only required for the "Developer Installation")
* Poetry

### Developer Installation with Poetry

Use these instructions to install the Intel AI Safety python library with a clone of the
GitHub repository. This can be done instead of the basic pip install, if you plan
on making code changes.

1. Clone this repo and navigate to the repo directory.

2. Allow poetry to create virtual envionment contained in `.venv` directory of current directory. 

   ```bash
   poetry lock
   ```
   In addtion, you can explicitly tell poetry which python instance to use
   
   ```bash
   poetry env use /full/path/to/python
   ```

3. Choose the `intel_ai_safety` subpackages and plugins that you wish to install.
   
   a. Install `intel_ai_safety` with all of its subpackages (e.g. `explainer` and `model_card_gen`) and plugins
   ```bash
   poetry install --extras all
   ```

   b. Install `intel_ai_safety` with just `explainer`
   ```bash
   poetry install --extras explainer
   ```
   
   c. Install `intel_ai_safety` with just `model_card_gen`
   ```bash
   poetry install --extras model-card
   ```
   
   d. Install `intel_ai_safety` with `explainer` and all of its plugins
   ```bash
   poetry install --extras explainer-all
   ```

   e. Install `intel_ai_safety` with `explainer` and just its pytorch implementations
   
   ```bash
   poetry install --extras explainer-pytorch
   ```
   
   f. Install `intel_ai_safety` with `explainer` and just its tensroflow implementations
   
   ```bash
   poetry install --extras explainer-tensorflow
   ``` 

4. Activate the environment:

   ```bash
   source .venv/bin/activate
   ```

### Install to existing enviornment with Poetry

#### Create and activate a Python3 virtual environment
We encourage you to use a python virtual environment (virtualenv or conda) for consistent package management.
There are two ways to do this:
1. Choose a virtual enviornment to use:
   a. Using `virtualenv`:
      ```bash
      python3 -m virtualenv xai_env
      source xai_env/bin/activate
      ```

   b. Or `conda`:
      ```bash
      conda create --name xai_env python=3.9
      conda activate xai_env
      ```
2. Install to current enviornment
   ```bash
   poetry config virtualenvs.create false && poetry install --extras all
   ```

### Additional Feature-Specific Steps
Notebooks may require additional dependencies listed in their associated documentation.

### Verify Installation

Verify that your installation was successful by using the following commands, which display the Explainer and Model Card Generator versions:
```bash
python -c "from intel_ai_safety.explainer import version; print(version.__version__)"
python -c "from intel_ai_safety.model_card_gen import version; print(version.__version__)"
```

## Running Notebooks

The following links have Jupyter* notebooks showing how to use the Explainer and Model Card Generator APIs in various ML domains and use cases:
* [Model Card Generator Notebooks](notebooks#model-card-generator-tutorial-notebooks)
* [Explainer Notebooks](notebooks#explainer-tutorial-notebooks)

## Support

The Intel Explainable AI Tools team tracks bugs and enhancement requests using
[GitHub issues](https://github.com/intelai/intel-xai-tools/issues). Before submitting a
suggestion or bug report, search the existing GitHub issues to see if your issue has already been reported.

*Other names and brands may be claimed as the property of others. [Trademarks](http://www.intel.com/content/www/us/en/legal/trademarks.html)

#### DISCLAIMER
These scripts are not intended for benchmarking Intel platforms. For any performance and/or benchmarking information on specific Intel platforms, visit https://www.intel.ai/blog.
 
Intel is committed to the respect of human rights and avoiding complicity in human rights abuses, a policy reflected in the Intel Global Human Rights Principles. Accordingly, by accessing the Intel material on this platform you agree that you will not use the material in a product or application that causes or contributes to a violation of an internationally recognized human right.
 
#### License
Intel® Explainable AI Tools is licensed under Apache License Version 2.0.
 
#### Datasets and Models
To the extent that any data, datasets, or models are referenced by Intel or accessed using tools or code on this site such data, datasets and models are provided by the third party indicated as the source of such content. Intel does not create the data, datasets, or models, provide a license to any third-party data, datasets, or models referenced, and does not warrant their accuracy or quality. By accessing such data, dataset(s) or model(s) you agree to the terms associated with that content and that your use complies with the applicable license. [DATASETS](DATASETS.md)

Intel expressly disclaims the accuracy, adequacy, or completeness of any data, datasets or models, and is not liable for any errors, omissions, or defects in such content, or for any reliance thereon. Intel also expressly disclaims any warranty of non-infringement with respect to such data, dataset(s), or model(s). Intel is not liable for any liability or damages relating to your use of such data, datasets, or models.
