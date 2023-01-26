from setuptools import setup

setup(
    name='explainer-explainers-cam-explainer',
    version='0.1',
    zip_safe=False,
    platforms='any',
    py_modules=['cam_explainer'],
    include_package_data=True,
    install_requires=[
        'charset-normalizer==2.1.1',
        'contourpy==1.0.6',
        'cycler==0.11.0',
        'fonttools==4.38.0',
        'grad-cam==1.4.6',
        'joblib==1.2.0',
        'kiwisolver==1.4.4',
        'matplotlib==3.6.2',
        'numpy==1.23.5',
        'nvidia-cublas-cu11==11.10.3.66',
        'nvidia-cuda-nvrtc-cu11==11.7.99',
        'nvidia-cuda-runtime-cu11==11.7.99',
        'nvidia-cudnn-cu11==8.5.0.96',
        'opencv-python==4.6.0.66',
        'packaging==22.0',
        'pillow==9.3.0',
        'pyparsing==3.0.9',
        'python-dateutil==2.8.2',
        'requests==2.28.1',
        'scikit-image==0.19.3',
        'scikit-learn==1.1.3',
        'scipy==1.9.3',
        'threadpoolctl==3.1.0',
        'torch==1.13.0',
        'torchvision==0.14.0',
        'tqdm==4.64.1',
        'ttach==0.0.3',
        'typing-extensions==4.4.0',
    ],
    entry_points={
        'explainer.explainers.cam_explainer': [
            'xgradcam = cam_explainer:xgradcam [model,targetLayer,targetClass,image,device]',
        ]
    }, 
    python_requires='>=3.9,<3.10'
)
