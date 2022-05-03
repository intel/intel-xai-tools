

docs::
	jupyter-book build --all docs/


build::
	pip install -U -r docs/requirements.txt


clean::
	rm -rf docs/_build
