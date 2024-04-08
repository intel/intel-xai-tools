# Creating Model Card for Toxic Comments Classification in Tensorflow
This notebook demonstrates how to generate Model Card for a [Tensorflow model](https://colab.research.google.com/github/google/eng-edu/blob/main/ml/pc/exercises/fairness_text_toxicity_part1.ipynb?utm_source=practicum-fairness&utm_campaign=colab-external&utm_medium=referral&utm_content=fairnessexercise1-colab#scrollTo=2z_xzJ40j9Q-) trained to classify toxic comments using the [CivilComments dataset](https://www.tensorflow.org/datasets/catalog/civil_comments) from the Tensorflow data hub. 

`toxicity-tfma-model-card.ipynb` performs the following steps:
1. Import dependencies 
2. Download and process the CivilComments Dataset from TensorFlow data hub.
    - Load the training and validation data.
    - Prepare feature map.
    - Parse training data and handle data imbalance in the training data.
3. Build the classification model based on dense feed-forward neural networks using Tensorflow Estimator .
4. Train the classification model.
5. Export the trained classification model in EvalSavedModel Format.
6. Generate Model Card with the help of Intel Model Card Generator
    - Load or write the config file.
    - Define the model card dictionary following the model card schema.
    - Generate Model Card using the generate function from the ModelCardGen class.


## Running the notebook

To run the `toxicity-tfma-model-card.ipynb`, install the following dependencies:
1. [Intel® Explainable AI](https://github.com/IntelAI/intel-xai-tools)
2. `! pip install tensorflow_hub`

## References
### _Dataset citations_
```
@misc{pavlopoulos2020toxicity,
    title={Toxicity Detection: Does Context Really Matter?},
    author={John Pavlopoulos and Jeffrey Sorensen and Lucas Dixon and Nithum Thain and Ion Androutsopoulos},
    year={2020}, eprint={2006.00998}, archivePrefix={arXiv}, primaryClass={cs.CL}
}

@article{DBLP:journals/corr/abs-1903-04561,
  author    = {Daniel Borkan and
               Lucas Dixon and
               Jeffrey Sorensen and
               Nithum Thain and
               Lucy Vasserman},
  title     = {Nuanced Metrics for Measuring Unintended Bias with Real Data for Text
               Classification},
  journal   = {CoRR},
  volume    = {abs/1903.04561},
  year      = {2019},
  url       = {http://arxiv.org/abs/1903.04561},
  archivePrefix = {arXiv},
  eprint    = {1903.04561},
  timestamp = {Sun, 31 Mar 2019 19:01:24 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1903-04561},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

@inproceedings{pavlopoulos-etal-2021-semeval,
    title = "{S}em{E}val-2021 Task 5: Toxic Spans Detection",
    author = "Pavlopoulos, John  and Sorensen, Jeffrey  and Laugier, L{'e}o and Androutsopoulos, Ion",
    booktitle = "Proceedings of the 15th International Workshop on Semantic Evaluation (SemEval-2021)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.semeval-1.6",
    doi = "10.18653/v1/2021.semeval-1.6",
    pages = "59--69",
}
