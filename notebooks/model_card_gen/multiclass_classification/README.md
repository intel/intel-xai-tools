# Creating Model Card for Multiclass Classification of Hate Speech
In this notebook, we will download a model, dataset, and metric from Hugging Face Hub and generate an interactive HTML Model Card using Intel AI Safety Model Card Generator Tool.

`multiclass-classification-model-card.ipynb` performs the following steps:
1. Download and Import Dependencies
2. Download the Dataset from Hugging Face Datasets
3. Transform the Dataset
   - Get ground truth labels
5. Download the Model and Process the Model Outputs
6. Load the Bias Metric form Hugging Face
7. Run the Bias Metric for each class label
8. Transform the Output for the Model Card for each class label
9. Build the Model Card


## Running the notebook

To run the `multiclass-classification-model-card.ipynb`, install the following dependencies:
1. [Intel® Explainable AI](https://github.com/Intel/intel-xai-tools)
2. `!pip install evaluate datasets transformers[torch] scikit-learn`

## References
### _Dataset citations_
```
@article{mathew2020hatexplain,
  title={HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection},
  author={Mathew, Binny and Saha, Punyajoy and Yimam, Seid Muhie and Biemann, Chris and Goyal, Pawan and Mukherjee, Animesh},
  journal={arXiv preprint arXiv:2012.10289},
  year={2020}
}
```