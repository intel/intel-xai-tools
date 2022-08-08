(design)=
# Explainer Design

<details>
<summary>Overview</summary>
<br/>

Explainer adheres to a set of principles, itemized below, based on software best practices and architectural design patterns. These guideposts allow explainer to keep pace with the rapid rate of innovation within XAI by keeping implementation agnostic. Explainer uses a data centric approach of passing data artifacts such as models, features and datasets into resource related entry points so that specific XAI implementations can be called. Explainer's CLI/API provides import/export mechanisms to bundle XAI functionality as python archives. The python archives are loaded at runtime either implicitly using python's import statement or explicitly using explainer's CLI/API. Details are provided in subsequent sections.

Ensure Intel Optimizations are included
: Any explainable method needs to leverage Intel libraries and configurations that provide optimizations.

Keep containers slim and lightweight
: Adding new explainations shouldn't require new containers to be built or should result in adding to the container size. Leverage explanations suites so that specific explainables can be included depending on the notebook or pipeline being run.

Don't forsake security
: An Explainer YAML that includes python dependencies as a URI allows the explainer component to be located locally in the container, on a local volume mount or in a registry. The URI would allows for these different locations to be specified.

Be portable
: Adding an explanation to an existing workflow shouldn't require a new virtual environment, or force the user to do pip installs in the current python environment since that environment is now changed and may be a shared environment across many workflows.

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

</details>

<details>
<summary>Explainer <b>Resources</b></summary>
<br/>

Explainer uses yaml files to locate explainable resources and optionally call their related entry points. Yaml files are located in an explainer package named explainer.explainer.
Within this package are yaml files that explainer uses to load and invoke specific XAI resources.  Running 

> explainer list

returns a list of yaml files, where each yaml file has attributes that specify what resource to load, along with optional attributes such as the entry_point to call once the resource is loaded. The first line in each yaml files is a YAML annotation that reifies the yaml file as an ExplainerSpec dataclass (see {{PyYaml}}). The ExplainerSpec dataclass has the following structure show below:

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

**Utilizing python's import machinery**

Python's {{PEP451}} (introduced in python-3.4) enhances the import mechanism to be extensible and secure by introducing a type called ModuleSpec that the import machinery instantiates whenever a new module is loaded. This PEP expanded the types of Loaders and MetaPathLoaders that are allowed. Directly importing resources such as yaml is leveraged by the XAI explainer. When a yaml file is imported, the explainer will dynamically inject explainable resources within the current python environemnt by using customized its Loader and MetaPathLoader classes.

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
        --- !ExplainerSpec
        name: zero shot learning
        plugin: zero_shot_learning.pkz
        dependencies:
        - shap==0.40.0
        - transformers==4.20.1
        - torch==1.12.0
        entry_point: |  
          # SHAP Explainer
          def entry_point(pipe, text):
            import shap
            print(f"Shap version used: {shap.__version__}")
            explainer = shap.Explainer(pipe)
            shap_values = explainer(text)
            prediction = pipe(text)
            print(f"Model predictions are: {prediction}")
            shap.plots.text(shap_values)
            # Let's visualize the feature importance towards the outcome - sports
            shap.plots.bar(shap_values[0,:,'sports'])

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
    ExplainerLoader->>ExplainerLoader: call entry_point 

```

</details>

<details>
<summary>Explainer <b>ModuleSpec</b></summary>
<br/>

```{eval-rst}

.. autoclass:: explainer.ExplainerModuleSpec
   :noindex:
   :members:
   :inherited-members:

```

</details>
<details>
<summary>Explainer <b>Loader</b></summary>
<br/>

```{eval-rst}

.. autoclass:: explainer.ExplainerLoader
   :noindex:
   :members:
   :inherited-members:

```

</details>
<details>
<summary>Explainer <b>MetaPathFinder</b></summary>
<br/>

```{eval-rst}

.. autoclass:: explainer.ExplainerMetaPathFinder
   :noindex:
   :members:
   :inherited-members:

```

</details>
<details>
<summary>Explainer <b>CLI</b></summary>
<br/>

```{eval-rst}

.. automodule:: explainer.cli
   :noindex:
   :members:

```

</details>
<details>
<summary>Explainer <b>API</b></summary>
<br/>


```{eval-rst}

.. automodule:: explainer.api
   :noindex:
   :members:

```

</details>
<br/>
