# Generating Model Card with PyTorch
This notebook demonstrates how to generate a model card for a PyTorch model using the Intel Model Card Generator. The PyTorch model is trained on a prediction task using the [Adult dataset](https://archive.ics.uci.edu/dataset/2/adult) from OpenML.

`adult-pytorch-model-card.ipynb` performs the following steps:
1. Import the dependencies 
2. Load the Adult Dataset from OpenML and preprocess the data.
3. Build the Multilayer Neural Network (NN) using PyTorch.
4. Train and save the Multilayer NN model.
5. Generate Model Card with Intel Model Card Generator
  - Load or write the config file.
  - Define the model card dictionary following the model card schema for the Multilayer NN.
  - Generate Model Card using generate function from the ModelCardGen class.
  - Export the generated Model Card to html format.



## Running the notebook

To run `adult-pytorch-model-card.ipynb`, install the following dependencies:
1. [Intel® Explainable AI](https://github.com/IntelAI/intel-xai-tools)


## References
### _Dataset citations_
Becker,Barry and Kohavi,Ronny. (1996). Adult. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20.


### _Publication citations_
Ron Kohavi, "Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid", Proceedings of the Second International Conference on Knowledge Discovery and Data Mining, 1996

'Simoudis, Evangelos, Jiawei Han, and Usama Fayyad. Proceedings of the second international conference on knowledge discovery & data mining. No. CONF-960830-. AAAI Press, Menlo Park, CA (United States), 1996.

Friedler, Sorelle A., et al. "A Comparative Study of Fairness-Enhancing Interventions in Machine Learning." Proceedings of the Conference on Fairness, Accountability, and Transparency, 2019, https://doi.org/10.1145/3287560.3287589.

Lahoti, Preethi, et al. "Fairness without demographics through adversarially reweighted learning." Advances in neural information processing systems 33 (2020): 728-740.
