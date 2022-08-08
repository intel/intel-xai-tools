(xai)=
# Explainable AI Concepts

<details>
<summary>What is Explainable AI?</summary>
<br/>

{{XAI}} (also known by the acronym XAI) is a methodology to provide information about a model, data or its features that can be understood by humans. Explainability techniques can be applied at almost any point within a model's training or inference workflow. As shown below, explainability and interpretability are tighly coupled. Depending on the algorithm being used, different approaches to add explainability are contingent on what is interpretable.

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

</details>

<details>
<summary>XAI Approaches</summary>
<br/>

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

</details>

<details>
<summary>Toolkits that work well with transformers</summary>
<br/>

**transformer-interpret**

This library{{TransformersInterpret}} adds an explainer to any HuggingFace transformer. The python package combines both HuggingFace {{Transformers}} and {{Captum}}. The choice of a model within the HuggingFace {{Transformers}} library is done by using {{AutoClasses}}. An example of the API is shown below:

> model = AutoModel.from_pretrained("bert-base-cased")


In this case, the pretrained model "bert-base-cased" will be downloaded from the HuggingFace model repo on huggingface.com, added to a local python class cache and imported into the current python environment. The type of framework used with the pretained model is determined by the path or an additional boolean parameter in the method of from_tf. The bert model returned from the method differs depending on whether PyTorch or TensorFlow.


**path-explain**

This library{{PathExplain}} explains machine learning and deep learning models models based on the author's paper{cite}`janizek2020explaining` and is well integrated with HuggingFace {{Transformers}} library. The library explains both feature attributions and feature interactions 

</details>
<br/>
