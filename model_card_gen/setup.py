from distutils.command import build
from setuptools import setup

REQUIRED_PACKAGES = [
    'absl-py>=0.9,<0.11',
    'semantic-version>=2.8.0,<3',
    'jinja2>=3,<4',
    'jsonschema>=3.2.0,<4',
    'intel-tensorflow',
    'tensorflow-model-analysis>=0.37.0,<0.39.0',
    'tensorflow-data-validation>=1.6.0,<1.8.0',
    'plotly>=3.8.1,<6',
    'dataclasses;python_version<"3.7"',
]

TEST_PACKAGES = [
    'pytest',
    'tensorflow-hub',
]

NOTEBOOK_PACKAGES = [
    'sklearn',
    'tfx',
    'tensorflow-hub',
    'tensorflow-transform'
]

EXTRAS = {
    'test': TEST_PACKAGES,
    'notebook': NOTEBOOK_PACKAGES,
}

# Get version from version module.
with open('model_card_gen/version.py') as fp:
  globals_dict = {}
  exec(fp.read(), globals_dict)  # pylint: disable=exec-used
__version__ = globals_dict['__version__']

with open('README.md', 'r', encoding='utf-8') as fh:
  _LONG_DESCRIPTION = fh.read()


setup(
    name='model-card-generator',
    version=__version__,
    description='Model Card Generator',
    long_description=_LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url='',
    author='',
    author_email='',
    packages=[
        'model_card_gen',
        'model_card_gen.docs',
        'model_card_gen.docs.examples',
        'model_card_gen.utils'
    ],
    package_data={
        'model_card_gen': ['schema/**/*.json', 'template/**/*.jinja'],
        'model_card_gen.docs.examples': ['docs/examples/**/*.html',
                                         'docs/examples/**/*.json'],
    },
    python_requires='>=3.6,<4',
    install_requires=REQUIRED_PACKAGES,
    tests_require=TEST_PACKAGES,
    extras_require=EXTRAS,
    # PyPI package labels.
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='',
    keywords='model card',
)
