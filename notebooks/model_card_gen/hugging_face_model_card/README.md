# Creating Model Card for Toxic Comments Classification in Tensorflow
In this notebook, we will download a model, dataset, and metric from Hugigng Face Hub and generate a interactive HTML Model Card using Intel AI Safety Model Card Generator Tool.

`hugging-face-model-card` performs the following steps:
1. Download and Import Dependencies
2. Download Dataset from Hugging Face Datasets
3. Transform Dataset
   - Get ground truth label
5. Download Modle and Process Outputs
6. Get Bias Metric form Hugging Face
7. Run Bias Metric
8. Transform Output for Model Card
9. Build Model Card


## Running the notebook

To run the `toxicity-tfma-model-card.ipynb`, install the following dependencies:
1. [Intel® Explainable AI](https://github.com/Intel/intel-xai-tools)
2. `!pip install evaluate datasets transformers[torch] scikit-learn`

## References
### _Dataset citations_
```
@article{aluru2020deep,
    title={Deep Learning Models for Multilingual Hate Speech Detection},
    author={Aluru, Sai Saket and Mathew, Binny and Saha, Punyajoy and Mukherjee, Animesh},
    journal={arXiv preprint arXiv:2004.06465},
    year={2020}
}
```
