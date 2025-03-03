# Generating Model Card for Multimodal Classification
This notebook demonstrates generating a model card for multimodal vision and text classification task using PyTorch and [Roberta](https://arxiv.org/abs/1907.11692) models. The model is trained on a [Hateful Memes Challenge dataset](https://huggingface.co/datasets/neuralcatcher/hateful_memes) created by Facebook AI to detect hate speech in multimodal memes. Finally, we use the Intel Model Card Generator for analyzing the performance of our model.

`multimodal-classification-model-card.ipynb` performs the following steps:
1. Import the dependencies.
2. Load the Hateful Meme Challenge dataset from Hugging Face.
3. Generate text features using the RoBERTa model and pre-process the images.
4. Build the PyTorch model for both image and text features.
5. Train and save the PyTorch model.
6. Evaluate model performance on the test dataset and process the evaluation output for the model card.
7. Generate Model Card with Intel Model Card Generator -
    - Define the model card dictionary following the model card schema for the Image Classification model.
    - Generate Model Card using generate function from the ModelCardGen class.
    - Export the generated Model Card to html format.


## Running the notebook

To run `multimodal-classification-model-card.ipynb`, install the following dependencies:
1. [Intel® Explainable AI](https://github.com/Intel/intel-xai-tools)


## References
### _Publication citations_
```
@misc{liu2019robertarobustlyoptimizedbert,
title={RoBERTa: A Robustly Optimized BERT Pretraining Approach}, 
author={Yinhan Liu and Myle Ott and Naman Goyal and Jingfei Du and Mandar Joshi and Danqi Chen and Omer Levy and Mike Lewis and Luke Zettlemoyer and Veselin Stoyanov},
year={2019},
eprint={1907.11692},
archivePrefix={arXiv},
primaryClass={cs.CL},
url={https://arxiv.org/abs/1907.11692}, 
}

@misc{kiela2021hatefulmemeschallengedetecting,
title={The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes}, 
author={Douwe Kiela and Hamed Firooz and Aravind Mohan and Vedanuj Goswami and Amanpreet Singh and Pratik Ringshia and Davide Testuggine},
year={2021},
eprint={2005.04790},
archivePrefix={arXiv},
primaryClass={cs.AI},
url={https://arxiv.org/abs/2005.04790}, 
}
```
