# Detecting Issues in Fairness by Generate Model Card from Tensorflow Estimators
This notebook demonstrates how to create a TFX pipeline (originally published by [Tensorflow Authors](https://github.com/tensorflow/fairness-indicators/blob/r0.38.0/g3doc/tutorials/Fairness_Indicators_Lineage_Case_Study.ipynb)) for creating a Proxy model trained for detecting issues in fairness using the [COMPAS dataset](https://www.propublica.org/datastore/dataset/compas-recidivism-risk-score-data-and-analysis). Further, the notebook also demonstrates on how to generate a model card for the trained Proxy model using the Intel Model Card Generator.

`compas-model-card-tfx.ipynb` performs the following steps:
1. Import the dependencies 
2. Load and preprocess the COMPAS Dataset in a csv format from Google Cloud.
3. Build the custom TFX pipeline:
    - Transformer
	    -  Prepare mapping between string feature keys and transformed feature operations. Fill the missing values.
    - Trainer
    	- The eval_input_receiver_fn will return the TensorFlow graph for parsing the raw untransformed features by applying the tf-transform preprocessing operators.
    	-  Construct the keras model.
    	-  trainer_fn will train the model for classifying the COMPAS dataset when called in the TFX pipeline.
    - Pipeline
    	- Compute statistics over the data for visualization and example validation.
	    - Perform data transformations and feature engineering in training and serving.
	    - Implement model based on the trainer_args given by the user.
	    - Compute evaluation statistics over the features of a model and perform quality validation of a candidate model.
4. Generate Model Card with Intel Model Card Generator
    - Load or write the config file.
    - Define the model card dictionary following the model card schema.
    - Generate Model Card using generate function from the ModelCardGen class.
  
## Running the notebook

To run the `toxicity-tfma-model-card.ipynb`, install the following dependencies:
1. [Intel® Explainable AI](https://github.com/IntelAI/intel-xai-tools)

## References      
### _Dataset citations_
COMPAS dataset - https://www.propublica.org/datastore/dataset/compas-recidivism-risk-score-data-and-analysis

### _Publication citations_
1.  Wadsworth, C., Vera, F., Piech, C. (2017). Achieving Fairness Through Adversarial Learning: an Application to Recidivism Prediction. https://arxiv.org/abs/1807.00199.

2.  Chouldechova, A., G’Sell, M., (2017). Fairer and more accurate, but for whom? https://arxiv.org/abs/1707.00046.

3.  Berk et al., (2017), Fairness in Criminal Justice Risk Assessments: The State of the Art, https://arxiv.org/abs/1703.09207.

