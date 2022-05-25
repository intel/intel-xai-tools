
clean::
	jupyter-book clean --all .

build:: clean
	jupyter-book config sphinx . && \
	sphinx-build . ./_build/html -b html

serve::
	python -m http.server --directory ./_build/html 9010


install::
	pip install -U -r ./requirements.txt

