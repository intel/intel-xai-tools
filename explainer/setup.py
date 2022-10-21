"""
XAI Tools, Explainer
"""
from setuptools import setup

dependencies = [
  'click~=8.1.3',
  'click_completion~=0.5.2',
  'pyyaml~=6.0',
  'urllib3[secure]~=1.26.11',
  'typing_extensions==4.3.0',
  'bump2version==1.0.1'
]

setup(
    name='intel-xai-tools',
    version='0.2.0',
    url='https://github.com/IntelAI/intel-xai-tools',
    license='BSD',
    author='IntelAI',
    author_email='kam.d.kasravi@intel.com',
    description='Explainer invokes an explainer given a model, dataset and features',
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
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
		python_requires='>=3.9'
)
