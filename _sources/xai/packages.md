(xaipackages)=
# Explainable AI packages that work well with transformers

**transformer-interpret**

This package{{TransformersInterpret}} adds an explainer to any HuggingFace transformer. The python package combines both HuggingFace {{Transformers}} and {{Captum}}. The choice of a model within the HuggingFace {{Transformers}} library is done by using {{AutoClasses}}. An example of the API is shown below:

> model = AutoModel.from_pretrained("bert-base-cased")


In this case, the pretrained model "bert-base-cased" will be downloaded from the HuggingFace model repo on huggingface.com, added to a local python class cache and imported into the current python environment. The type of framework used with the pretained model is determined by the path or an additional boolean parameter in the method of from_tf. The bert model returned from the method differs depending on whether PyTorch or TensorFlow.


**path-explain**

This package{{PathExplain}} explains machine learning and deep learning models models based on the author's paper{cite}`janizek2020explaining` and is well integrated with HuggingFace {{Transformers}} library. This package explains both feature attributions and feature interactions 

</details>
<br/>
