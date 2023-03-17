# Documentation

## Sphinx Documentation

Install `intel-xai-tools` and it's dependencies for developers as described [here](/README.md##build-and-install)
```bash
# Run these commands from intel-xai-tools project folder
python3 -m virtualenv xai_env
source xai_env/bin/activate
python -m pip install --editable .
```

Install Sphinx and a few other tools required to build docs
```bash
pip install -r docs/requirements-docs.txt
```

Finally generate the html docs (from within `docs` directory):
```bash
make clean html
```

The output HTML files will be located in `docs/_build/html`.

To start a local HTTP server and view the docs locally, try:
```bash
make serve
Serving HTTP on 127.0.1.1 port 9999 (http://127.0.1.1:9999/) ...
```

If you need to view the docs from another machine, please try either port forwarding or
provide appropriate values for `LISTEN_IP/LISTEN_PORT` arguments.
For example:
```bash
LISTEN_IP=0.0.0.0 make serve
Serving HTTP on 0.0.0.0 port 9999 (http://0.0.0.0:9999/) ...
```

runs the docs server on the host while listening to all hosts.
Now you can navigate to `HOSTNAME:9999` to view the docs.