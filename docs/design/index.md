(design)=
# Design

## Overview

Explainer implements a plugin architecture in order to accommodate a wide variety of explainer types which work with different platforms (pytorch, tensorflow). The mechanism of importing leverages {{PEP451}} which defines a ModuleSpec type and how the python interpreter imports python code and resources. The explainer CLI provides a way to import and export a given explainer so that arbitrary explainers can be injected into a pipeline. 

### Optimizations

By using a customized loader, explainer can insure disparite explanations are added to underlying python environments that have all the necessary optimized libraries. 


### Scalability

Isolating different explainers by plugin allows a large number of explainers to be included in workflows without ballooning the underlying container.


### Security

An Explainer YAML that includes python dependencies as a URI allows the explainer component to be located locally in the container, on a local volume mount or in a registry. The URI would allows for these different locations to be specified.


### Portability

The explainer CLI provides both import and export subcommands so that development of a given explainer can be decoupled from using the explainer. 


### Version Compatibilities


Because there is a strong likelyhood that different explainers may require different versions of python packages, isolating these package dependencies within the explainer plugin avoids runtime errors related to version incompatabilities.

## Plugin Design

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
.. include:: ./cli.rst
```


```{eval-rst}
.. include:: ./api.rst
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

* {{YANG202229}}
* {{ZHU202253}}
* {{HOLZINGER202128}}

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
    classDef leafName fill:#00f,color:#fff;
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



## References

<details>
<summary>misc references</summary>

* [Logic Tensor Networks](https://github.com/logictensornetworks/logictensornetworks)
* [COUNTERFACTUAL EXPLANATIONS WITHOUT OPENING THE BLACK BOX: AUTOMATED DECISIONS AND THE GDPR](https://arxiv.org/pdf/1711.00399.pdf)
* [Generating Counterfactual Explanations with Natural Language](https://arxiv.org/pdf/1806.09809.pdf)

</details>
