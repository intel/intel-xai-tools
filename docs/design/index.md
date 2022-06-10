(design)=
# Design

## Overview

{{XAI}} is a methodology to provide information about a model, data or its features that can be understood by humans. Explainability techniques can be applied at almost any point within a model's training or inference workflow. As shown below, explainability and interpretability are tighly coupled. Depending on the algorithm being used, different approaches to add explainability are contingent on what is interpretable.

```{figure} ../images/explain1.png
---
name: interpretability-methods
---
Various interpretability methods
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
    C -->|Yes| D["Logic\nTensor\nNetworks\n(<b>LTN</b>)"]
    C -->|No| G{CounterFactual\nExplanation?}
    E -->|Yes| F["Transformer\nInterpret"]
    E -->|No| J{"Image Data?"}
    J -->|Yes| K["Gradient-weighted\nClass\nActivation\nMapping\n(<b>Grad-CAM)</b>"]
    J -->|No| L["Layer-wise\nRelevance\nPropagation\n(<b>LDP</b>)"]
    G -->|Yes| H["Diverse\nCounterfactual\nExplanations\n(<b>DICE</b>) fa:fa-external-link-alt"]
    G -->|No| I["Shapley\nAdditive\nExplanations\n(<b>SHAP</b>)"]
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

Given the rapid availability of new explainability techniques as well as how different explainations can be inserted throughout a model's lifecycle for both training and inference, the challenge is how does one go about providing new explainable techniques within workflows? Additionally, how does one keep up with the myriad of explainable techniques and remain compatible with existing workflows that have already been published? This seems difficult if not impossible functionality for a toolkit to provide especially in light of hard requirements noted below:

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


## Python's Plugin Architecture


Python's {{PEP451}}, introduced in python-3.4 transformed a brittle and error prone import mechanism to one that is extensible and secure. Moreover, it introduced a type called ModuleSpec that the import machinery instantiates whenever a new module is loaded. Finally, {{PEP451}} expanded the types of Loaders and MetaPathLoaders that are allowed. The way to find and load python classes and resources such as csv's was outlined specifically to provide new extensibility architectures, known as plugin architectures. As noted in the article {{PluginArchitectureinPython}}:


> At its core, a plugin architecture consists of two components: a core system and plug-in modules. The main key design here is to allow adding additional features that are called plugins modules to our core system, providing extensibility, flexibility, and isolation to our application features. This will provide us with the ability to add, remove, and change the behaviour of the application with little or no effect on the core system or other plug-in modules making our code very modular and extensible .


This architecture is specifically geared to meet the functionality required by XAI explainations. Injecting explanations within a workflow is synonomous with providing "extensibility, flexibility, and isolation to our application features". Yet there are many ways to do this, specifically what approach makes sense for the common use cases that add explanations?  The upcoming {{PEP690}}, due in python-3.12 further enhances how imports work, by allowing imports to be lazily loaded.

### How is an explanation typically added to a notebook?



## Existing Approaches to Add Explainations

Explainer implements a plugin architecture in order to accommodate a wide variety of explainer types which work with different platforms (pytorch, tensorflow). The mechanism of importing leverages {{PEP451}} which defines a ModuleSpec type and how the python interpreter imports python code and resources. The explainer CLI provides a way to import and export a given explainer so that arbitrary explainers can be injected into a pipeline. 

The native python plugin architecture provides a way to add specific functionality to a framework at runtime. In this case there are many different types of explainers that need to be added to a general workflow framework. Explainer uses python's Loader so that different explainable implementations can be loaded into the current environment.
It does so by loading python code and resources defined in a YAML file.


Explainer adds a customized Loader and MetaPathLoader class shown below so that YAML files are imported. These YAML files leverage {{PyYaml}} to do customized loading.


```{mermaid}
:caption: "ExplainerSpec"

classDiagram
    class ExplainerSpec
    ExplainerSpec: +String name
    ExplainerSpec: +String data 
    ExplainerSpec: +String entry_point
    ExplainerSpec: +String model
    ExplainerSpec: +String plugin

```

```{eval-rst}

.. autoclasstree:: explainer.ExplainerLoader
   :caption: Class Hierarchy of explainer.ExplainerLoader
   :full:

```

```{eval-rst}

.. autoclasstree:: explainer.ExplainerMetaPathFinder
   :caption: Class Hierarchy of explainer.ExplainerMetaPathFinder
   :full:

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

```{eval-rst}

.. click:: explainer:cli
  :prog: explainer
  :nested: full

```


```{eval-rst}
.. automodule:: explainer.api
   :noindex:
   :members:

```

## Algorithms and Data Flows

<details>
<summary>Algorithms</summary>


```{mermaid}
:caption: "Explainability based on Algorithms{cite}`chou2022counterfactuals`"

%%{
  init: { "flowchart": { "htmlLabels": true, "curve": "linear" } }
}%%

flowchart LR
    A[Algorithm] --> B{Is\nyour\nmodel\ninterpretable?}
    B -->|Yes| C[Use\nIntrinsic\nmethods]
    B -->|No| D{Explain\nindividual\npredictions\nor\nentire\nmodel?}
    D -->|Entire Model| F{"Does\nyour\nmodel\nhave\na\nstandard\narchitecture?"}
    D -->|Individual Predictions| J{"Does\nyour\nmodel\nhave\na\nstandard\narchitecture?"}
    D -->|Both| E
    F -->|No| K["Model\nagnostic\nmethods\nlike\nPartial\nDependence\nplots"]
    F -->|Yes| L["Use\nModel\nspecific\nglobal\nmethods\nlike\nXGBoost"]
    J -->|Yes| M["Model\nspecific\nlocal\nmethods\nlike\nGrad-CAM</b>"]
    J -->|No| E
    E["SHAP\nor\nLIME"]
    classDef leafName fill:#00f,color:#fff;
    class C,E,K,L,M leafName;
```

</details>

<details>
<summary>Data Flows</summary>


```{mermaid}
:caption: "Explainable Data{cite}`bennetot2021practical`"

%%{
  init: { "flowchart": { "htmlLabels": true, "curve": "linear" } }
}%%

flowchart LR
    A[Data] --> B{Tabular Data?}
    B -->|Yes| C{Interactive\nExplanation?}
    B -->|No| E{Text Data?}
    C -->|Yes| D["Logic\nTensor\nNetworks\n(<b>LTN</b>)"]
    C -->|No| G{CounterFactual\nExplanation?}
    E -->|Yes| F["Transformer\nInterpret"]
    E -->|No| J{"Image Data?"}
    J -->|Yes| K["Gradient-weighted\nClass\nActivation\nMapping\n(<b>Grad-CAM)</b>"]
    J -->|No| L["Layer-wise\nRelevance\nPropagation\n(<b>LDP</b>)"]
    G -->|Yes| H["Diverse\nCounterfactual\nExplanations\n(<b>DICE</b>) fa:fa-external-link-alt"]
    G -->|No| I["Shapley\nAdditive\nExplanations\n(<b>SHAP</b>)"]
    classDef leafName fill:#eee;
    class D,F,H,I,K,L leafName;
```

* Logic Tensor Networks: See {cite}`bennetot2021practical`
* See {cite}`logictensornetworks`
* See {cite}`mothilal2020explaining`

</details>

<details>
<summary>State-of-the-art approaches integrating explanations into workflows</summary>

### transformer-interpret and path-explain

transformer-interpret
: This library{{TransformersInterpret}} adds an explainer to any HuggingFace transformer. The python package combines both HuggingFace {{Transformers}} and {{Captum}}. The choice of a model within the HuggingFace {{Transformers}} library is done by using {{AutoClasses}}. An example of the API is shown below:

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

path-explain
: This library{{PathExplain}} adds an explainer that can also accept either a PyTorch or TensorFlow model. The library explains feature importances and feature interactions in deep neural networks using path attribution methods.


```{eval-rst}

.. autoclasstree:: path_explain.explainers.embedding_explainer_tf.EmbeddingExplainerTF
   :caption: Class Hierarchy of path_explain.explainers.embedding_explainer_tf.EmbeddingExplainerTF
   :full:

```

</details>

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
