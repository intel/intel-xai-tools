# Design

## Overview


### Automating the selection of Models

Auto Classes
: The choice of a model within the huggingface transformers library is done by using [AutoClasses](https://huggingface.co/docs/transformers/main/en/model_doc/auto#auto-classes). For example the BertModel has the hierarchy shown below, where Module is a PyTorch Module.


```{eval-rst}

.. autoclasstree:: transformers.models.bert.BertModel
   :caption: Class Hierarchy of transformers.models.bert.BertModel
   :full:

```




generate counterfactuals

What explainers are available is registered under explainers.

* IntegratedGradients(model)
* NoiseTunnel(ig)
* DeepLift(model)
* GradientShap(model)
* FeatureAblation(model)

<details>
<summary>Explainability Approach based on Data</summary>


```{mermaid}
:caption: "Explainability based on Data"

%%{
  init: { "flowchart": { "htmlLabels": true, "curve": "linear" } }
}%%

flowchart TD
    A[Start] --> B{Tabular Data?}
    B -->|Yes| C{Interactive\nExplanation?}
    B -->|No| E{Text Data?}
    C -->|Yes| D["Logic Tensor Networks"]
    C -->|No| G{CounterFactual\nExplanation?}
    E -->|Yes| F["Transformer Interpret"]
    E -->|No| J["Image Data?"]
    J -->|Yes| K["Grad Cam"]
    J -->|No| L["LDP"]
    G -->|Yes| H["DICE"]
    G -->|No| I["SHAP"]
```

* Logic Tensor Networks: See {cite}`bennetot2021practical`
* See {cite}`logictensornetworks`
* See {cite}`mothilal2020explaining`

</details>

<details>
<summary>Explainability Approach based on Models</summary>

```{mermaid}
flowchart LR
    A[Hard edge] -->|Link text| B(Round edge)
    B --> C{Decision}
    C -->|One| D[Result one]
    C -->|Two| E[Result two]
```

</details>

<details>
<summary>Explainer API</summary>

```{eval-rst}
.. include:: ./api.rst
```

</details>

<details>
<summary>Explainer CLI</summary>

```{eval-rst}
.. include:: ./cli.rst
```

</details>

## References:

* [Logic Tensor Networks](https://github.com/logictensornetworks/logictensornetworks)
* [COUNTERFACTUAL EXPLANATIONS WITHOUT OPENING THE BLACK BOX: AUTOMATED DECISIONS AND THE GDPR](https://arxiv.org/pdf/1711.00399.pdf)
* [Generating Counterfactual Explanations with Natural Language](https://arxiv.org/pdf/1806.09809.pdf)
* [transformers-interpret](https://github.com/cdpierse/transformers-interpret)
* [path-explain](https://github.com/suinleelab/path_explain)


```{mermaid}
flowchart TD
    B["fa:fa-twitter for peace"]
    B-->C[fa:fa-ban forbidden]
    B-->D(fa:fa-spinner);
    B-->E(A fa:fa-camera-retro perhaps?)
```
