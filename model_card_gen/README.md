# Intel® Explainable AI Tools

## Model Card Generator

Model Card Generator allows users to create interactive HTML reports of containing model performance and fairness metrics

**Model Card Sections**

<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky">Section<br></th>
    <th class="tg-0pky">Subsection</th>
    <th class="tg-73oq">Decription</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky" rowspan="9">Model Details</td>
    <td class="tg-0pky">Overview</td>
    <td class="tg-0pky">A brief, one-line description of the model card.</td>
  </tr>
  <tr>
    <td class="tg-0pky">Documentation</td>
    <td class="tg-0pky">A thorough description of the model and its usage.</td>
  </tr>
  <tr>
    <td class="tg-0pky">Owners</td>
    <td class="tg-0pky">The individuals or teams who own the model.</td>
  </tr>
  <tr>
    <td class="tg-0pky">Version</td>
    <td class="tg-0pky">The version of the schema</td>
  </tr>
  <tr>
    <td class="tg-0pky">Licenses</td>
    <td class="tg-0pky">The model's license for use.</td>
  </tr>
  <tr>
    <td class="tg-0pky">References</td>
    <td class="tg-0pky">Links providing more information about the model.</td>
  </tr>
  <tr>
    <td class="tg-0pky">Citations</td>
    <td class="tg-0pky">How to reference this model card.</td>
  </tr>
  <tr>
    <td class="tg-0pky">Path</td>
    <td class="tg-0pky">The path where the model is stored.</td>
  </tr>
  <tr>
    <td class="tg-0pky">Graphics</td>
    <td class="tg-0pky">Collection of overview graphics.</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="6">Model Parameters</td>
    <td class="tg-0pky">Model Architecture</td>
    <td class="tg-0pky">The architecture of the model.</td>
  </tr>
  <tr>
    <td class="tg-0pky">Data</td>
    <td class="tg-0pky">The datasets used to train and evaluate the model.</td>
  </tr>
  <tr>
    <td class="tg-0pky">Input Format</td>
    <td class="tg-0pky">The data format for inputs to the model.</td>
  </tr>
  <tr>
    <td class="tg-0pky">Input Format Map</td>
    <td class="tg-0pky">The data format for inputs to the model, in key-value format.</td>
  </tr>
  <tr>
    <td class="tg-0pky">Output Format</td>
    <td class="tg-0pky">The data format for outputs from the model.</td>
  </tr>
  <tr>
    <td class="tg-0pky">Output Format Map</td>
    <td class="tg-0pky">The data format for outputs from the model, in key-value format.</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="2">Quantitative analysis</td>
    <td class="tg-0pky">Performance Metrics</td>
    <td class="tg-0pky">The model performance metrics being reported.</td>
  </tr>
  <tr>
    <td class="tg-0pky">Graphics</td>
    <td class="tg-0pky">Colleciton of performance graphics</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="5">Considerations</td>
    <td class="tg-0pky">Users</td>
    <td class="tg-0pky">Who are the intended users of the model?</td>
  </tr>
  <tr>
    <td class="tg-0pky">Use Cases</td>
    <td class="tg-0pky">What are the intended use cases of the model?</td>
  </tr>
  <tr>
    <td class="tg-0pky">Limitations</td>
    <td class="tg-0pky">What are the known technical limitations of the model? E.g. What kind(s) of data should the model be expected not to perform well on? What are the factors that might degrade model performance?</td>
  </tr>
  <tr>
    <td class="tg-0pky">Tradeoffs</td>
    <td class="tg-0pky">What are the known tradeoffs in accuracy/performance of the model?</td>
  </tr>
  <tr>
    <td class="tg-0pky">Ethical Considerations</td>
    <td class="tg-0pky">What are the ethical (or environmental) risks involved in the application of this model?</td>
  </tr>
</tbody>
</table>

### Intall

Step 1: Clone repo

```shell
git clone https://github.com/IntelAI/intel-xai-tools.git
```

Step 2: Navigate to `model-card-generator` package

```shell
cd intel-xai-tools/model_card_gen
```

Step 3: Intall with pip

```shell
pip install .
```

For notebook dependencies install with

```shell
pip install ".[notebook]"
```


### Run

**Populate Model Card user-defined fields**
```python
mc = {
  "model_details": {
    "name": "COMPAS (Correctional Offender Management Profiling for Alternative Sanctions)",
    "overview": "COMPAS (Correctional Offender Management Profiling for Alternative Sanctions) is a public dataset, which contains approximately 18,000 criminal cases from Broward County, Florida between January, 2013 and December, 2014. The data contains information about 11,000 unique defendants, including criminal history demographics, and a risk score intended to represent the defendant’s likelihood of reoffending (recidivism)",
    "owners": [
      {
        "name": "Intel XAI Team",
        "contact": "xai@intel.com"
      }
    ],
    "references": [
      {
        "reference": "Wadsworth, C., Vera, F., Piech, C. (2017). Achieving Fairness Through Adversarial Learning: an Application to Recidivism Prediction. https://arxiv.org/abs/1807.00199."
      },
      {
        "reference": "Chouldechova, A., G'Sell, M., (2017). Fairer and more accurate, but for whom? https://arxiv.org/abs/1707.00046."
      },
      {
        "reference": "Berk et al., (2017), Fairness in Criminal Justice Risk Assessments: The State of the Art, https://arxiv.org/abs/1703.09207."
      }
    ],
    "graphics": {
      "description": " "
    }
  },
  "quantitative_analysis": {
    "graphics": {
      "description": " "
    }
  },
  "schema_version": "0.0.1"
}
```

**Define Inputs**
```python
_project_path = os.path.join('examples', 'compas')
_model_path = os.path.join(_project_path, 'model')
_data_paths = {'eval': os.path.join(_project_path, 'eval.tfrecord'),
               'train': os.path.join(_project_path, 'train.tfrecord')}
_eval_config = os.path.join(_project_path, 'eval_config.proto')
```

**Create Model Card**
```python
from model_card_gen.model_card_gen import ModelCardGen
mcg = ModelCardGen.generate(_data_paths, _model_path, _eval_config, model_card=mc)
```

### Test

Step 1: Test by installing test dependencies:

```shell
pip install ".[test]"
```

Step 2: Run tests

```shell
python -m pytest model_card_gen/tests/
```
