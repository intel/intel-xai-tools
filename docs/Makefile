
html::
	sphinx-build --keep-going . ./_build/html -b html

clean::
	rm -rf _build

publish::
	ghp-import -n -p -f _build/html

serve::
	python -m http.server --directory ./_build/html 9010

install::
	pip install -U -r ./requirements.txt

autodoc::
	sphinx-apidoc -o /tmp explainer
