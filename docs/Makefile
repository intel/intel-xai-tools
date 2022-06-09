
clean::
	rm -f conf.py && jupyter-book clean --all .

<<<<<<< HEAD:docs/Makefile
html::
	jupyter-book config sphinx . && \
	sphinx-build . ./_build/html -b html
	#jupyter-book build . --all
=======
html:: clean
	#jupyter-book config sphinx . && \
	#sphinx-build . ./_build/html -b html
	jupyter-book build . --all
>>>>>>> 93ca392c0f2d33055b6cfdee18c632103f39668e:explainer/docs/Makefile

serve::
	python -m http.server --directory ./_build/html 9010

install::
	pip install -U -r ./requirements.txt

autodoc::
	sphinx-apidoc -o /tmp explainer
