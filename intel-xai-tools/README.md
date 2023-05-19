# Intel® Explainable AI Tools

This repository provides tools for data scientists and MLOps engineers that have requirements specific to AI model interpretability.

## Features
| Core Feature | Description | 
|----------|-----------|
| [Model Card Generator](model_card_gen) |  **Allows users to create interactive HTML reports containing model performance and fairness metrics.** |
|[Explainer](explainer) | **Allows users to run post-hoc model distillation and visualization methods to examine predictive behavior for both TensorFlow and PyTorch models via a simple Python API including the following modules:** <li> [Attributions](explainer/attributions/): visualize negative and positive attributions of tabular features, pixels, and word tokens for predictions <li> [CAM](explainer/cam/): create heatmaps for CNN image classifications using gradient-weight class activation CAM mapping <li> [Metrics](explainer/metrics/): Gain insight into models with the measurements and visualizations needed during the machine learning workflow|

## Build and Install
Requirements:
* Linux system (or WSL2 on Windows)
* git
* required python version: 3.9
* `apt-get install build-essential python3-dev libgl1 libglib2.0-0`

### Basic Installation:
```
pip install intel-xai
```
### Advanced/Developer Installation:
1. Clone this repo and navigate to the repo directory:
   ```
   git clone https://github.com/IntelAI/intel-xai-tools.git

   cd intel-xai-tools/intel-xai-tools
   ```
2. Create and activate a Python3.9 virtual environment using `virtualenv`:
   ```
   python3.9 -m virtualenv xai_env
   source xai_env/bin/activate
   ```

   Or `conda`:
   ```
   conda create --name xai_env python=3.9
   conda activate xai_env
   ```
3. Install this tool with 
   ```
   make install
   ```

## Running Notebooks

Run [example notebooks](../notebooks) that show how to use the explainer and model card generator API in various ML domains and use cases. Notebooks may require additional dependencies listed in their associated README's.

### DISCLAIMER: ###
These scripts are not intended for benchmarking Intel platforms. For any performance and/or benchmarking information on specific Intel platforms, visit https://www.intel.ai/blog.
 
Intel is committed to the respect of human rights and avoiding complicity in human rights abuses, a policy reflected in the Intel Global Human Rights Principles. Accordingly, by accessing the Intel material on this platform you agree that you will not use the material in a product or application that causes or contributes to a violation of an internationally recognized human right.
 
### License: ###
Intel® Explainable AI Tools is licensed under Apache License Version 2.0.
 
#### Datasets: ###
To the extent that any public datasets are referenced by Intel or accessed using tools or code on this site those datasets are provided by the third party indicated as the data source. Intel does not create the data, or datasets, and does not warrant their accuracy or quality. By accessing the public dataset(s) you agree to the terms associated with those datasets and that your use complies with the applicable license. [DATASETS](DATASETS.md)
 
Intel expressly disclaims the accuracy, adequacy, or completeness of any public datasets, and is not liable for any errors, omissions, or defects in the data, or for any reliance on the data.  Intel is not liable for any liability or damages relating to your use of public datasets.
