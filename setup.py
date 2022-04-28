"""
XAI Tools, Explainer
"""
from setuptools import find_packages, setup

dependencies = ['click', 'shap']

setup(
    name='explainable_ai_tooling',
    version='0.1.0',
    url='https://github.com/kkasravi/explainable-ai-tooling',
    license='BSD',
    author='Kam Kasravi',
    author_email='kam.d.kasravi@intel.com',
    description='XAI Tools',
    long_description=__doc__,
    py_modules=["explainer", "explainer.cli"],
    include_package_data=True,
    zip_safe=False,
    platforms='any',
    install_requires=dependencies,
    entry_points={
        'console_scripts': [
            'explainer = explainer.cli:cli',
        ],
    },
    classifiers=[
        # As from http://pypi.python.org/pypi?%3Aaction=list_classifiers
        # 'Development Status :: 1 - Planning',
        # 'Development Status :: 2 - Pre-Alpha',
        # 'Development Status :: 3 - Alpha',
        'Development Status :: 4 - Beta',
        # 'Development Status :: 5 - Production/Stable',
        # 'Development Status :: 6 - Mature',
        # 'Development Status :: 7 - Inactive',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX',
        'Operating System :: MacOS',
        'Operating System :: Unix',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ]
)
