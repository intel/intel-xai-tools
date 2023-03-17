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

# [Explaining Text Classification](https://coderzcolumn.com/tutorials/artificial-intelligence/explain-text-classification-models-using-shap-values-keras)

```{code-cell} ipython3
from explainer.explainers import feature_attributions_explainer, metrics_explainer
```

```{code-cell} ipython3
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['KMP_WARNINGS'] = 'off'

import numpy as np
from sklearn import datasets

all_categories = ['alt.atheism','comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware',
                  'comp.sys.mac.hardware','comp.windows.x', 'misc.forsale','rec.autos','rec.motorcycles',
                  'rec.sport.baseball','rec.sport.hockey','sci.crypt','sci.electronics','sci.med',
                  'sci.space','soc.religion.christian','talk.politics.guns','talk.politics.mideast',
                  'talk.politics.misc','talk.religion.misc']

selected_categories = ['alt.atheism','comp.graphics','rec.motorcycles','sci.space','talk.politics.misc']

X_train_text, Y_train = datasets.fetch_20newsgroups(subset="train", categories=selected_categories, return_X_y=True)
X_test_text , Y_test  = datasets.fetch_20newsgroups(subset="test", categories=selected_categories, return_X_y=True)

X_train_text = np.array(X_train_text)
X_test_text = np.array(X_test_text)

classes = np.unique(Y_train)
mapping = dict(zip(classes, selected_categories))

len(X_train_text), len(X_test_text), classes, mapping
```

```{code-cell} ipython3
print(Y_test)
```

## Vectorize Text Data

```{code-cell} ipython3
import sklearn
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=50000)

vectorizer.fit(np.concatenate((X_train_text, X_test_text)))
X_train = vectorizer.transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

X_train, X_test = X_train.toarray(), X_test.toarray()

X_train.shape, X_test.shape
```

## Define the Model

```{code-cell} ipython3
:tags: [remove-output]

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

def create_model():
    return Sequential([
                        layers.Input(shape=X_train.shape[1:]),
                        layers.Dense(128, activation="relu"),
                        layers.Dense(64, activation="relu"),
                        layers.Dense(len(classes), activation="softmax"),
                    ])

model = create_model()

```

```{code-cell} ipython3
model.summary()
```

## Compile and Train Model

```{code-cell} ipython3
:tags: [remove-output]

model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(X_train, Y_train, batch_size=256, epochs=5, validation_data=(X_test, Y_test))
```

## Evaluate Model Performance

```{code-cell} ipython3
:tags: [hide-output]

from sklearn.metrics import accuracy_score, classification_report

train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

print("Train Accuracy : {:.3f}".format(accuracy_score(Y_train, np.argmax(train_preds, axis=1))))
print("Test  Accuracy : {:.3f}".format(accuracy_score(Y_test, np.argmax(test_preds, axis=1))))
print("\nClassification Report : ")
print(classification_report(Y_test, np.argmax(test_preds, axis=1), target_names=selected_categories))
```

```{code-cell} ipython3
# one-hot-encode clasess
oh_Y_test = np.eye(len(classes))[Y_test]

cm = metrics_explainer['confusionmatrix'](oh_Y_test, test_preds, selected_categories)
cm.visualize()
print(cm.report)
```

```{code-cell} ipython3
plotter = metrics_explainer['plot'](oh_Y_test, test_preds, selected_categories)
plotter.pr_curve()
```

```{code-cell} ipython3
plotter.roc_curve()
```

```{code-cell} ipython3
import re

X_batch_text = X_test_text[1:3]
X_batch = X_test[1:3]

print("Samples : ")
for text in X_batch_text:
    print(re.split(r"\W+", text))
    print()

preds_proba = model.predict(X_batch)
preds = preds_proba.argmax(axis=1)

print("Actual    Target Values : {}".format([selected_categories[target] for target in Y_test[1:3]]))
print("Predicted Target Values : {}".format([selected_categories[target] for target in preds]))
print("Predicted Probabilities : {}".format(preds_proba.max(axis=1)))
```

## SHAP Partition Explainer

+++

## Visualize SHAP Values Correct Predictions

```{code-cell} ipython3
:tags: [remove-output]

def make_predictions(X_batch_text):
    X_batch = vectorizer.transform(X_batch_text).toarray()
    preds = model.predict(X_batch)
    return preds

partition_explainer = feature_attributions_explainer.partitionexplainer(make_predictions, r"\W+", selected_categories)(X_batch_text)
```

### Text Plot

```{code-cell} ipython3
partition_explainer.visualize()
```

### Bar Plots

+++

#### Bar Plot 1

```{code-cell} ipython3
shap = partition_explainer.shap
shap_values = partition_explainer.shap_values

shap.plots.bar(partition_explainer.shap_values[:,:, selected_categories[preds[0]]].mean(axis=0), max_display=15,
               order=shap.Explanation.argsort.flip)
```

#### Bar Plot 2

```{code-cell} ipython3
shap.plots.bar(shap_values[0,:, selected_categories[preds[0]]], max_display=15,
               order=shap.Explanation.argsort.flip)
```

### Bar Plot 3

```{code-cell} ipython3
shap.plots.bar(shap_values[:,:, selected_categories[preds[1]]].mean(axis=0), max_display=15,
               order=shap.Explanation.argsort.flip)
```

### Bar Plot 4

```{code-cell} ipython3
shap.plots.bar(shap_values[1,:, selected_categories[preds[1]]], max_display=15,
               order=shap.Explanation.argsort.flip)
```

## Waterfall Plots

+++

### Waterfall Plot 1

```{code-cell} ipython3
shap.waterfall_plot(shap_values[0][:, selected_categories[preds[0]]], max_display=15)
```

### Waterfall Plot 2

```{code-cell} ipython3
shap.waterfall_plot(shap_values[1][:, selected_categories[preds[1]]], max_display=15)
```

## Force Plot

```{code-cell} ipython3
import re
tokens = re.split("\W+", X_batch_text[0].lower())
shap.initjs()
shap.force_plot(shap_values.base_values[0][preds[0]], shap_values[0][:, preds[0]].values,
                feature_names = tokens[:-1], out_names=selected_categories[preds[0]])
```

```{code-cell} ipython3

```
