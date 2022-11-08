from setuptools import setup

setup(
    name='explainer-explainers-metrics_explainer',
    version='0.1',
    zip_safe=False,
    platforms='any',
    py_modules=['metrics_explainer'],
    include_package_data=True,
    install_requires=[
        'matplotlib==3.6.0',
        'seaborn==0.12.0',
        'scikit-learn==1.1.2',
        'pandas==1.5.0',
        'plotly==5.10.0',
        'jupyter-plotly-dash==0.4.3',
    ],
    entry_points={ 
        'explainer.explainers.metrics_explainer': [
            'confusionmatrix= metrics_explainer:confusion_matrix [groundtruth,predictions,labels]',
            'plot=metrics_explainer:plot [groundtruth,predictions,labels]',
            'pstats=metrics_explainer:pstats [command]',
        ]
    }, 
    python_requires='>=3.9,<3.10'
)
