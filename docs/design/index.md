(design)=
# Design

## Overview

{{XAI}} is a methodology to provide information about a model, data or its features that can be understood by humans. Explainability techniques can be applied at almost any point within a model's training or inference workflow. As shown below, explainability and interpretability are tighly coupled. Depending on the algorithm being used, different approaches to add explainability are contingent on what is interpretable.

```{figure} ../images/explain1.png
---
name: interpretability-methods
---
Various interpretability methods {cite}`e23010018`
```

Viewed as more of a state diagram, types of explanations are shown in the figure below, where the state is shown as a decision tree. For example, if the model is already interpretable then there are many metrics available that can show the accuracy of the prediction or classification. On the other hand, if the model is too complex to be interpretable such as deep learning models, then different options exist depending on what the user is exploring.


```{figure} ../images/explain5.png
---
name: explainable-algos
---
Explaination based on Algorithm
```

Based on data alone, a similar state diagram is shown below, where a data-centric approach is used that leverages explainations that are tuned to image or text. Based on the model topology, these types of explanations would be added to object detection or nlp models. 

```{mermaid}
:caption: "Explainable Data{cite}`bennetot2021practical`"

%%{
  init: { "flowchart": { "htmlLabels": true, "curve": "linear" } }
}%%

flowchart LR
    A[Data] --> B{Tabular Data?}
    B -->|Yes| C{Interactive\nExplanation?}
    B -->|No| E{Text Data?}
    C -->|Yes| D["Logic\nTensor\nNetworks\n(LTN)"]
    C -->|No| G{CounterFactual\nExplanation?}
    E -->|Yes| F["Transformer\nInterpret"]
    E -->|No| J{"Image Data?"}
    J -->|Yes| K["Gradient-weighted\nClass\nActivation\nMapping\n(Grad-CAM)"]
    J -->|No| L["Layer-wise\nRelevance\nPropagation\n(LDP)"]
    G -->|Yes| H["Diverse\nCounterfactual\nExplanations\n(DICE)"]
    G -->|No| I["Shapley\nAdditive\nExplanations\n(SHAP)"]
    classDef leafName fill:#eee;
    class D,F,H,I,K,L leafName;
```


### Explosion of Explainable Approaches

The types of approaches that can be used to better explain a model as shown below are growing.

```{figure} ../images/explain2.png
---
name: explainable-approaches
---
Various approaches to explainability

```

As the table below shows, explainable toolkits abound and are growing rapidly.

```{figure} ../images/explain4.png
---
name: explainable-toolkits
---
Various Toolkits available from 2021

```


## Toolkit Strategy Constraints

Given the rapid availability of new explainability techniques as well as how different explainations can be inserted throughout a model's lifecycle for both training and inference, the challenge is how does one go about providing new explainable techniques within workflows? Additionally, how does one keep up with the myriad of explainable techniques and remain compatible with existing workflows that have already been published? This seems difficult if not impossible functionality for a toolkit to provide especially in light of functional requirements noted below:

Ensure Intel Optimizations are included
: Any explainable method needs to leverage Intel libraries and configurations that provide optimizations.

Keep containers slim and lightweight
: Adding new explainations shouldn't require new containers to be built or should result in adding to the container size. A large suite of explainations should be available where specific ones can be included depending on the notebooks being shown. 

Don't forsake security
: An Explainer YAML that includes python dependencies as a URI allows the explainer component to be located locally in the container, on a local volume mount or in a registry. The URI would allows for these different locations to be specified.

Be portable
: Adding an explaination to an existing workflow shouldn't require a new virtual environment, or pip installs in a current environment since that virtual environment is now changed and may be a shared environment across many workflows.

Be repeatable
: An explanation that has dependencies on the model, data or features should ensure that these dependencies are version compatible.

Do not wrap native APIs
: Providing a wrapper around an existing XAI toolkit does not scale

Do not mandate a particular platform (tensorflow, pytorch, etc)
: Explainable techniques and methods that are specific to a platform should be filtered out when that platform is not in the workflow

Do not mandate a type of model
: Explainable techniques and methods that expect a specific model class should be filtered out when that class is not available

Do not mandate a type of data
: Explainable techniques and methods that expect a data format class should be filtered out when that format is not available

## Existing Approaches to Add Explainations

### Explicit/Manual Approach (typical)

In the example cells below, taken from a typical ML workflow that includes shap explanations, the shap module is installed into the current jupyter kernel, then imported into the notebook along with other imports.

```python

#!pip install shap

```

```python
import numpy as np
import matplotlib
# SHAP explanation
import shap
```

Next, the notebook will load a dataset and split the dataset into train and test partitions ...

```python
boston = datasets.load_boston()
X_train, X_test, y_train, y_test = model_selection.train_test_split(boston.data, boston.target, random_state=0)
```

After the data part of the pipeline, the model is created and trained on the trained data split


```python
regressor = ensemble.RandomForestRegressor()
regressor.fit(X_train, y_train);
```

Post training, the notebook creates an explainer, passing it the model and trained data split.


```python
explainer = shap.TreeExplainer(regressor)
# Calculate Shap values
shap_values = explainer.shap_values(X_train)
```

### State-of-the-art Approaches

#### transformer-interpret

This library{{TransformersInterpret}} adds an explainer to any HuggingFace transformer. The python package combines both HuggingFace {{Transformers}} and {{Captum}}. The choice of a model within the HuggingFace {{Transformers}} library is done by using {{AutoClasses}}. An example of the API is shown below:

> model = AutoModel.from_pretrained("bert-base-cased")


In this case, the pretrained model "bert-base-cased" will be downloaded from the HuggingFace model repo on huggingface.co, added to a local python class cache and imported into the current python environment. The type of framework used with the pretained model is determined by the path or an additional boolean parameter in the method of from_tf. The bert model returned from the method differs depending on whether PyTorch or TensorFlow is used (see figures below).


```{eval-rst}

.. autoclasstree:: transformers.AutoModelForSequenceClassification
   :caption: Class Hierarchy of transformers.AutoModelForSequenceClassification
   :full:

```

```{eval-rst}

.. autoclasstree:: transformers.models.bert.BertModel
   :caption: Class Hierarchy of transformers.models.bert.BertModel for pytorch
   :full:

```

```{eval-rst}

.. autoclasstree:: transformers.models.bert.TFBertModel
   :caption: Class Hierarchy of transformers.models.bert.TFBertModel for tensorflow
   :full:

```

## Python's importlib


Python's {{PEP451}}, introduced in python-3.4 enhanced the import mechanism to be extensible and secure by introducing a type called ModuleSpec that the import machinery instantiates whenever a new module is loaded. This PEP expanded the types of Loaders and MetaPathLoaders that are allowed. Directly importing resources such as yaml is leveraged by the XAI explainer. When a yaml file is imported, the explainer will dynamically inject explanations within a workflow as described below.


### Explainer and Explainable Resources

#### Explainer implicit injection of an Explainable Resource

By leveraging the python import machinery, explainer can implicitly load and import an explainable resource. Explainer adds customized Loader and MetaPathLoader classes noted below. 

```{eval-rst}

.. autoclasstree:: explainer.ExplainerLoader
   :caption: Class Hierarchy of explainer.ExplainerLoader
   :full:

.. autoclasstree:: explainer.ExplainerMetaPathFinder
   :caption: Class Hierarchy of explainer.ExplainerMetaPathFinder
   :full:

```

These classes are called when python resolves imports. As described in {{MetaPathFinders}}, a yaml file can be directly loaded by the import machinary so that the following import statement:

```python
from explainer.explainers import zero_shot_learning
```

resolves to a yaml file named zero_shot_learing.yaml (rather than a python file) located in the explainer.explainers package. This yaml file is shown below:


```{card}
:class-card: sd-text-black, sd-bg-light
zero_shot_learning.yaml
^^^
--- !ExplainerSpec<br/>
name: zero shot learning<br/>
plugin:zero_shot_learning.zip<br/>
dependencies:
- shap==0.40.0
```

The first line is a YAML annotation that uses {{PyYaml}} to reify the yaml file as an ExplainerSpec dataclass, shown below.

```{mermaid}
:caption: "ExplainerSpec"

classDiagram
    class ExplainerSpec
    ExplainerSpec: +String name
    ExplainerSpec: +String dataset
    ExplainerSpec: +List dependencies
    ExplainerSpec: +String entry_point
    ExplainerSpec: +String model
    ExplainerSpec: +String plugin

```

The set of steps that implicitly injects an explainable resource are shown in the sequence diagram below:


```{mermaid}
:caption: "Explainer sequence diagram when resolving a yaml file"

sequenceDiagram
    participant ExplainerLoader
    participant zero_shot_learning.yaml
    participant ExplainerSpec
    ExplainerLoader->>zero_shot_learning.yaml: find yaml file
    zero_shot_learning.yaml->>ExplainerLoader: load yaml file
    ExplainerLoader->>ExplainerSpec: create
    ExplainerSpec->>ExplainerLoader: fields initialized from yaml file
    ExplainerLoader->>zero_shot_learning.zip: find zip file
    zero_shot_learning.zip->>ExplainerLoader: extract zip file
    ExplainerLoader->>ExplainerLoader: add path to sys.path

```


```{eval-rst}

.. autoclass:: explainer.ExplainerLoader
   :noindex:
   :members:
   :inherited-members:


.. autoclass:: explainer.ExplainerMetaPathFinder
   :noindex:
   :members:
   :inherited-members:

```

#### Explainer explicit injection of an Explainable Resource (CLI/API)

The Explainer CLI provides export and import subcommands. Each command requires a path to an explainable yaml file. The CLI will instantiate the Explainer API and call its export_to or import_from, passing in the yaml path. This is then delegated to the ExplainerLoader, which then follows the sequence diagram noted in the implicit use case above.


```{eval-rst}
.. automodule:: explainer.cli
   :noindex:
   :members:

```


```{eval-rst}
.. automodule:: explainer.api
   :noindex:
   :members:

```

* Logic Tensor Networks: See {cite}`bennetot2021practical`
* See {cite}`logictensornetworks`
* See {cite}`mothilal2020explaining`
* {{YANG202229}}
* {{ZHU202253}}
* {{HOLZINGER202128}}


## References

<details>
<summary>misc references</summary>

* [Logic Tensor Networks](https://github.com/logictensornetworks/logictensornetworks)
* [COUNTERFACTUAL EXPLANATIONS WITHOUT OPENING THE BLACK BOX: AUTOMATED DECISIONS AND THE GDPR](https://arxiv.org/pdf/1711.00399.pdf)
* [Generating Counterfactual Explanations with Natural Language](https://arxiv.org/pdf/1806.09809.pdf)

</details>
