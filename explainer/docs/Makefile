
clean::
	jupyter-book clean --all .

html::
	jupyter-book config sphinx . && \
	sphinx-build . ./_build/html -b html

serve::
	python -m http.server --directory ./_build/html 9010

install::
	pip install -U -r ./requirements.txt

autodoc::
	sphinx-apidoc -o /tmp explainer
