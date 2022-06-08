# Design

## Overview

Explainer implements a plugin architecture in order to accommodate a wide variety of explainer types which work with different platforms (pytorch, tensorflow). The mechanism of importing leverages PEP 451 which defines a ModuleSpec type and how the python interpreter imports python code and resources. The explainer CLI provides a way to import and export a given explainer so that arbitrary explainers can be injected into a pipeline. 


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

## Explainer Components

<details>
<summary>Plugin Design</summary>

Python Plugins
: A plugin package is a collection of related plugins corresponding to a Python package. An example is {{Glue}}

The native python plugin architecture provides a way to add specific functionality to a framework that is required to be extensible. In this case there are many explainers that needs to be added to a general workflow framework. 


Explainer uses python's Loader so that different explainable implementations can be loaded into the current environment.
It does so by just-in-time loading of python dependencies and explainable inputs that are defined in a yaml file. 


from explainer.explainers import noisetunnel

PyYaml
: The plugin architecture can be combined with {{PyYaml}} so that imports of yaml files can do customized loading

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
.. include:: ./plugin.rst
```

</details>

<details>
<summary>CLI Design</summary>

```{eval-rst}
.. include:: ./cli.rst
```

</details>

<details>
<summary>API Design</summary>

```{eval-rst}
.. include:: ./api.rst
```

</details>


## References

<details>
<summary>misc references</summary>

* [Logic Tensor Networks](https://github.com/logictensornetworks/logictensornetworks)
* [COUNTERFACTUAL EXPLANATIONS WITHOUT OPENING THE BLACK BOX: AUTOMATED DECISIONS AND THE GDPR](https://arxiv.org/pdf/1711.00399.pdf)
* [Generating Counterfactual Explanations with Natural Language](https://arxiv.org/pdf/1806.09809.pdf)

</details>
