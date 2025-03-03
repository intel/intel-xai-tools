# Generating Model Card for a Image Classification task
This notebook demonstrates how to generate a model card for an image classification task for a pretrained [ResNet50 torchvision model](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html)  with the Intel Model Card Generator. The ResNet model is finetuned on the [ HuggingFace dataset](https://huggingface.co/datasets/itsLeen/deepfake_vs_real_image_detection) for classifying images into either Deepfake or Normal images.

`image-classification-model-card.ipynb` performs the following steps:
1. Import the dependencies .
2. Load the Deepfake v/s Real Image Dataset from HuggingFace and preprocess the data.
3. Load the pre-trained Torchvision ResNet model.
4. Fine-tune and save the ResNet model.
5. Evaluate model performance on test dataset.
6. Generate Model Card with Intel Model Card Generator -
    - Define the model card dictionary following the model card schema for the Image Classification model.
    - Generate Model Card using generate function from the ModelCardGen class.
    - Export the generated Model Card to html format.


## Running the notebook

To run `image-classification-model-card.ipynb`, install the following dependencies:
1. [Intel® Explainable AI](https://github.com/Intel/intel-xai-tools)


## References
### _Publication citations_
```
@misc{he2015deepresiduallearningimage,
      title={Deep Residual Learning for Image Recognition}, 
      author={Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
      year={2015},
      eprint={1512.03385},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1512.03385}, 
}
```
