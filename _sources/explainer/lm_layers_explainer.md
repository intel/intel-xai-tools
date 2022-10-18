---
file_format: mystnb
kernelspec:
  name: python3
---
# Language Models

```{code-cell} python3
:tags: [remove-input]
from explainer.explainers import lm_layers_explainer
```

Today transformer models provide SOTA performance across NLP and CV fields. Transformer model variants can be causal language models (clm), masked language models (mlm) or encoder-decoder language models (enc-dec). This set of functions explores, visualizes and interacts with transformer based language models. 
Specifically:

- explain individual predictions by visualizing input token importance
- explain hidden state contributions and evolution across model layers
- visualize sequence embeddings 
- visualize attention heads within the self-attention layer


## Algorithms

Transformer based language models take words as input, turn them into tokens 
and provide numeric scores for each token in the model's vocabulary. 
A model's final layer, called the softmax layer, provides a probability score 
for each token. Transformer based architectures are comprised of 2 major layers:
- the self-attention layer
- the feed forward neural network layer


## Environment
- jupyter notebooks


## XAI Methods
- Gradient saliency
- Input X Gradient saliency
- Integrated Gradients
- DeepLift
- DeepLife with SHAP
- Guided Backpropagation
- Guided GradCam
- Deconvolution
- Layer-wise Relevance Propagation
- Gradient-Based Saliency


## Toolkits
- ecco
- bertviz


## References

- [Visualize BERT sequence embeddings: An unseen way](https://towardsdatascience.com/visualize-bert-sequence-embeddings-an-unseen-way-1d6a351e4568)
- [Visualize attention in NLP Models](https://github.com/jessevig/bertviz)
- [Interfaces for Explaining Transformer Language Models](https://jalammar.github.io/explaining-transformers/)
- [Attention is not not Explanation](https://arxiv.org/pdf/1908.04626.pdf?ref=morioh.com&utm_source=morioh.com)
- [Attention is not Explanation](https://arxiv.org/abs/1902.10186?ref=morioh.com&utm_source=morioh.com)
- [Attention Interpretability Across NLP Tasks](https://arxiv.org/pdf/1909.11218.pdf?ref=morioh.com&utm_source=morioh.com)


```{eval-rst}

.. automodule:: lm_layers_explainer
   :noindex:
   :members:

```
