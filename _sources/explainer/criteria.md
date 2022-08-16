(criteria)=
# Criteria

Ensure Intel Optimizations are included
: Any explainable method needs to leverage Intel libraries and configurations that provide optimizations.

Do not complicate or pollute python environments
: Adding new explanations shouldn't require new packages to be added to the existing python environment that may potentially corrupt or conflict with resident packages. Adding an explanation also shouldn't require a new virtual environment, or change the existing environment in ways that are difficult to reverse.

Don't forsake security
: An Explainer YAML that includes python dependencies as a URI allows the explainer component to be located locally in the container, on a local volume mount or in a registry. The URI would allows for these different locations to be specified.

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
