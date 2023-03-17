# Multimodal Breast Cancer Detection Explainability using the Intel® Explainable AI  API

This notebook demonstrate how to use the Explainable AI API for PyTorch CNN and Transformer models in a multimodal architecture. This notebook also includes the Intel® Transfer Learning Tool and the Intel® Extension for Transformers library.

`Multimodal_Cancer_Detection.ipynb` performs the following steps:
1. Import dependencies
2. Prepare image and text datasets
3. CNN image classification
    1. Analyze image dataset
    2. Get the model
    3. Download and prepare the dataset
    4. Transfer learning
    5. Evaluate
    6. Error Analysis
    7. Explanation
4. BERT text classification
    1. Analyze text dataset
    2. Get the model
    3. Prepare the dataset
    4. Transfer learning
    5. Evaluate
    6. Error Analysis
    7. Explanation
5.  Post-training quantization
    1. Configure and quantize the current BERT model
    2. Evaluate
    3. Error Analysis
6.  Ensemble
    1. Map images to text examples
    2. Execute ensemble algorithm
    3. Error analysis
    4. Explanation

The `dataset_utils.py` holds the supporting functions that prepare the image and text datasets.

## Running the notebook

To run `Multimodal_Cancer_Detection.ipynb`, install the following dependencies:
1. [Intel® Explainable AI](https://github.com/IntelAI/intel-xai-tools)
2. [Intel® Transfer Learning Tool](https://github.com/IntelAI/transfer-learning)
3. `pip install intel-extension-for-transformers`
4. `pip install scikit-image`
5. `pip install jupyterlab`
6. `pip install jupyter-dash`
7. `pip install nltk`
8. `pip install docx2txt`
9. `pip install openpyxl`

## References

### _Dataset citations_
Khaled R., Helal M., Alfarghaly O., Mokhtar O., Elkorany A., El Kassas H., Fahmy A. <b>Categorized Digital Database for Low energy and Subtracted Contrast Enhanced Spectral Mammography images [Dataset].</b> (2021) The Cancer Imaging Archive. DOI:  [10.7937/29kw-ae92](https://doi.org/10.7937/29kw-ae92)

### _Publication Citation_
Khaled, R., Helal, M., Alfarghaly, O., Mokhtar, O., Elkorany, A., El Kassas, H., & Fahmy, A. <b>Categorized contrast enhanced mammography dataset for diagnostic and artificial intelligence research.</b> (2022) Scientific Data, Volume 9, Issue 1. DOI: [10.1038/s41597-022-01238-0](https://doi.org/10.1038/s41597-022-01238-0)

### _TCIA Citation_
Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, Moore S, Phillips S, Maffitt D, Pringle M, Tarbox L, Prior F. <b>The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository</b>, Journal of Digital Imaging, Volume 26, Number 6, December, 2013, pp 1045-1057. DOI: [10.1007/s10278-013-9622-7](https://doi.org/10.1007/s10278-013-9622-7)