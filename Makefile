
explainer::
	pip install -U -e .

autodoc::
	sphinx-apidoc -o /tmp explainer

docs-clean::
	jupyter-book clean --all docs

docs:: docs-clean
	jupyter-book build --all docs/

docs-serve::
	python -m http.server --directory docs/_build/html 9009


docs-build::
	pip install -U -r docs/requirements.txt

