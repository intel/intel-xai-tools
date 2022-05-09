
explainer::
	pip install -e .

docs::
	jupyter-book build --all docs/


serve-docs::


build::
	pip install -U -r docs/requirements.txt


clean::
	rm -rf docs/_build
