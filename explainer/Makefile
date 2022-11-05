DOCS_DIR ?= ../docs
explainer::
	pip install -e .

explainer-whl::
	python setup.py bdist_wheel

clean::
	@rm -rf build dist intel_xai_tools.egg-info

test::
	pytest tests

bump-version::
	pip install bump2version==1.0.1
	bump2version --current-version 0.2.0 minor explainer/version.py

docs-install::
	cd ${DOCS_DIR} && make install

docs::
	cd ${DOCS_DIR} && make html

docs-publish::
	ghp-import -n -p -f ${DOCS_DIR}/_build/html

LISTEN_IP ?= 127.0.0.1
LISTEN_PORT ?= 9090
docs-serve::
	@echo serve docs locally on ${LISTEN_IP}:${LISTEN_PORT}
	@python -m http.server --directory ${DOCS_DIR}/_build/html ${LISTEN_PORT} --bind ${LISTEN_IP}
