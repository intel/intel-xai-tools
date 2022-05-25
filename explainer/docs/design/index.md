---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Design

## Overview

<details open>
<summary>Explainability Flow Diagram</summary>

The choice of explainability algorithms depends on the model as well as the data{cite}`YANG202229`. 

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

<details open>
<summary>Explainability Approaches based on Algorithms</summary>

```{mermaid}
:caption: "Explainability based on Algorithms{cite}`chou2022counterfactuals`"

%%{
  init: { "flowchart": { "htmlLabels": true, "curve": "linear" } }
}%%

flowchart LR
    A[Algorithm] --> B[Instance\nCentric]
    A[Algorithm] --> C[Constraint\nCentric]
    A[Algorithm] --> D[Genetic\nCentric]
    A[Algorithm] --> E[Regression\nCentric]
    A[Algorithm] --> F[Game Theory\nCentric]
    A[Algorithm] --> G[Case-based\nCentric]
    A[Algorithm] --> H[Probabilistic\nCentric]
    classDef leafName fill:#00f,color:#fff;
    class B,C,D,E,F,G,H leafName;
```

</details>

<details open>
<summary>Leveraging the python plugin architecture

Explainer provides an extension to python's Loader that allows explainable implementations to be loaded into the current environment.
It does so by just-in-time loading of python dependencies and explainable inputs that are within a yaml file. 

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

</details>



How would one add an explainer to a Jupyter Notebook or CI/CD pipeline? Below are several github repositories where this is done:

* {{TransformersInterpret}} - adds an explainer to a hugging face model
* {{PathExplain}} - adds an explainers that can either accept Pytorch or TensorFlow models.

The transformer-interpret library leverages both Captum and HuggingFace


Python Plugins
: A plugin package is a collection of related plugins corresponding to a Python package. An example is {{Glue}}

The plugin architecture provides a way to add specific functionality to a framework that is required to be extensible. In this case there are many explainers that needs to be added to a general explainer framework. 


PyYaml
: The plugin architecture can be combined with {{PyYaml}} so that imports of yaml files can do customized loading


### Automating the selection of Models

Auto Classes
: The choice of a model within the huggingface transformers library is done by using {{AutoClasses}} For example the BertModel has the hierarchy shown below, where Module is a PyTorch Module.


```{eval-rst}

.. autoclasstree:: transformers.models.bert.BertModel
   :caption: Class Hierarchy of transformers.models.bert.BertModel
   :full:

```

```{eval-rst}

.. autoclasstree:: transformers.models.bert.TFBertModel
   :caption: Class Hierarchy of transformers.models.bert.TFBertModel
   :full:

```


### Extending python's ModuleSpec to accomodate explainer dependencies, dataclasses and features.


generate counterfactuals

What explainers are available is registered under explainers.

* IntegratedGradients(model)
* NoiseTunnel(ig)
* DeepLift(model)
* GradientShap(model)
* FeatureAblation(model)

<details open>
<summary>Explainer Plugin Classes</summary>

```{eval-rst}
.. include:: ./plugin.rst
```

</details>

<details open>
<summary>Explainer API</summary>

```{eval-rst}
.. include:: ./api.rst
```

</details>

<details open>
<summary>Explainer CLI</summary>

```{eval-rst}
.. include:: ./cli.rst
```

</details>

## References:

* [Logic Tensor Networks](https://github.com/logictensornetworks/logictensornetworks)
* [COUNTERFACTUAL EXPLANATIONS WITHOUT OPENING THE BLACK BOX: AUTOMATED DECISIONS AND THE GDPR](https://arxiv.org/pdf/1711.00399.pdf)
* [Generating Counterfactual Explanations with Natural Language](https://arxiv.org/pdf/1806.09809.pdf)

