from setuptools import setup

setup(
    name='explainer-explainers-feature-attributions-explainer',
    version='0.1',
    zip_safe=False,
    platforms='any',
    py_modules=['feature_attributions_explainer'],
    include_package_data=True,
    install_requires=[
        'intel-tensorflow==2.9.1',
        'intel-scipy==1.7.3',
        'captum==0.5.0',
        'shap @ git+https://github.com/slundberg/shap@v0.41.0',
        'scikit-learn==1.1.2',
        'scikit-plot==0.3.7',
        'transformers==4.20.1',
        'torch==1.12.0',
        'opencv-python==4.6.0.66',
    ],
    entry_points={ 
        'explainer.explainers.feature_attributions_explainer': [
            'kernelexplainer = feature_attributions_explainer:kernel_explainer [model,data]',
            'deepexplainer = feature_attributions_explainer:deep_explainer [model,backgroundImages,targetImages,labels]',
            'gradientexplainer = feature_attributions_explainer:gradient_explainer',
            'partitionexplainer = feature_attributions_explainer:partition_explainer [task,model,data]',
            'integratedgradients = feature_attributions_explainer:integratedgradients [model]',
            'deeplift = feature_attributions_explainer:deeplift [model]',
            'smoothgrad = feature_attributions_explainer:smoothgrad [model]',
            'featureablation = feature_attributions_explainer:featureablation [model]',
            'saliency = feature_attributions_explainer:saliency [model]',
            'sentiment_analysis = feature_attributions_explainer:sentiment_analysis [model, text]',
        ]
    }, 
    python_requires='>=3.9'
)
