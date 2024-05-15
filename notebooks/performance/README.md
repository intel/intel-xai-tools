# Introduction
These two notebooks, together, compare the performance of `PartitionExplainer()` when toggling two binary experiment parameters:
1. Intel® oneDNN flags
2. AI Tools Environment (AI Tools)

Thus, these notebooks are able to cover the following four user environments

| |Intel®  oneDNN OFF | Intel® oneDNN ON |
| -- | :--: | :--: |
| __Stock Environment__ | X | X |
| __AI Tools Environment__ | X | X |

A "stock environment" is one where all of the Python packages installed from either PyPi's or Conda's default channels. The AI Tools environment, on the other hand, installs all Python packages from Conda's Intel channel, if available. Otherwise, all packages that are not available on the Intel Conda channel are then installed via Conda's default channel.

## Notebook Descriptions
### __partition_image_bm.ipynb__
- executes the  `PartitionExplainer()` twice - once with oneDNN flags set to off and a second time with the OneDNN flags set to on
- Scales each experiment by increasing the number of the `max_evals` shap parameter
- compares the performance of the two experiments 
- outputs results folders holding pickle files of the raw computation times and csv files holding the aggregated results used for visualization
### __aitools_vs_stock.ipynb__
- compares and visualizes two results folders output from the `partition_image_bm.ipynb` notebook
- it is designed such that it expects one of the results folders is from a experiment ran in a stock environment and the other in an AI Tools environment so that the user can compare the two environments
- no `PartitionExplaienr()` execution is actually done. It is only to compare experiments from `partition_image_bm.ipynb`

# Environments
In order to compare AI Tools and stock environments, two isolated Conda environments must be created.

### __Stock Conda environment__
```bash
conda create -n stock python=3.9
conda activate stock
pip install intel-xai --no-deps
pip install tensorflow==2.14.0 torch=2.2.0 ipywidgets notebook opencv-python shap
```

### __AI Tools Conda environment__
```bash
conda create -n intel -c intel python=3.9
conda activate intel
conda install -c intel --deps-only shap
conda install --no-deps shap
pip install --no-deps intel-xai
conda install -c intel pytorch=2.2.0 tensorflow=2.14.0 ipywidgets matplotlib notebook opencv
```
