---
jupytext:
  formats: ipynb,md:myst
  orphan: true
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Explaining Language Model Activations

+++

## See the [ecco toolkit](https://ecco.readthedocs.io/en/main/){cite}`alammar-2021-ecco`

```{code-cell} ipython3
from explainer.explainers import lm_layers_explainer

import warnings
warnings.filterwarnings('ignore')
```

```{code-cell} ipython3
lm_layers_explainer.entry_points
```

## Show activations in each layer in the model

```{code-cell} ipython3
model = 'bert-base-uncased'
text = ''' 
Now I ask you: what can be expected of man since he is a being endowed with strange qualities? 
Shower upon him every earthly blessing, drown him in a sea of happiness, so that nothing but bubbles of bliss 
can be seen on the surface; give him economic prosperity, such that he should have nothing else to do but sleep, 
eat cakes and busy himself with the continuation of his species, and even then out of sheer ingratitude, sheer spite, 
man would play you some nasty trick. He would even risk his cakes and would deliberately desire the most fatal rubbish, 
the most uneconomical absurdity, simply to introduce into all this positive good sense his fatal fantastic element. 
It is just his fantastic dreams, his vulgar folly that he will desire to retain, simply in order to prove to himself--as though that were so necessary-- 
that men still are men and not the keys of a piano, which the laws of nature threaten to control so completely that soon one will be able to desire nothing but by the calendar. 
And that is not all: even if man really were nothing but a piano-key, even if this were proved to him by natural science and mathematics, even then he would not become reasonable,
but would purposely do something perverse out of simple ingratitude, simply to gain his point. And if he does not find means he will contrive destruction and chaos, will 
contrive sufferings of all sorts, only to gain his point! He will launch a curse upon the world, and as only man can curse (it is his privilege, the primary distinction 
between him and other animals), may be by his curse alone he will attain his object--that is, convince himself that he is a man and not a piano-key!
'''
activations = lm_layers_explainer['activations'](model, text)
activations(n_components=8)
```

## Fill in the blank: "Heathrow airport is located in the city of __"

```{code-cell} ipython3
model = 'distilgpt2'
text = " Heathrow airport is in the city of"
predictions = lm_layers_explainer['predictions'](model, text)
```

## Visualize the candidate tokens at the last layer of the model (layer 5)

```{code-cell} ipython3
predictions(position=8, layer=5)
```

## We can see more tokens using the topk parameter

```{code-cell} ipython3
predictions(position=8, layer=5, topk=20)
```

## Visualize the candidate tokens at every layer

```{code-cell} ipython3
predictions(position=8)
```

## Show rankings for the next word

```{code-cell} ipython3
model = 'distilgpt2'
text = "The keys to the cabinet"
rankings = lm_layers_explainer['rankings'](model)
rankings(text, generate=1, do_sample=False).rankings_watch(watch=[318, 389], position=5)
```

## Attention head view

### Usage
- Hover over any token on the left/right side of the visualization to filter attention from/to that token. The colors correspond to different attention heads.
- Double-click on any of the colored tiles at the top to filter to the corresponding attention head.
- Single-click on any of the colored tiles to toggle selection of the corresponding attention head.
- Click on the Layer drop-down to change the model layer (zero-indexed).
- The lines show the attention from each token (left) to every other token (right). Darker lines indicate higher attention weights. When multiple heads are selected, the attention weights are overlaid on one another.

```{code-cell} ipython3
model = 'bert-base-uncased'
text = "The cat sat on the mat"
print(lm_layers_explainer['attention_head_view'].__doc__)
head_view = lm_layers_explainer['attention_head_view'](model)

head_view(text)
```

```{code-cell} ipython3
model = 'bert-base-uncased'
sentence_a = "The cat sat on the mat"
sentence_b = "The cat lay on the rug"
neuron_view = lm_layers_explainer['attention_neuron_view'](model)
neuron_view(sentence_a, sentence_b)
```

```{code-cell} ipython3
model = 'facebook/bart-large-cnn'
sentence_a = "The House Budget Committee voted Saturday to pass a $3.5 trillion spending bill"
sentence_b = "The House Budget Committee passed a spending bill."
model_view = lm_layers_explainer['attention_model_view'](model)
model_view(sentence_a, sentence_b)
```
