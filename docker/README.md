## Intel® Explainable AI Tools Dockerfiles and images
Currently there are Dockerfiles for both `Model Card Generator` and `Explainers` and they are based on `ubuntu:22.04`.
If you plan to build these containers from scratch, please pull the latest base first:
```bash
docker pull ubuntu:22.04
```

* [Explainers](#explainers)
* [Model Card Generator](#model-card-generator)
* [Model Card Generator UI](#model-card-generator-ui)
* [Interactive](#interactive)
* [Jupyter](#jupyter)

## Explainers

### Build the image
```bash
docker compose build explainers
```

### Check existing image:
```bash
docker images | grep -i explainers
intel/ai-tools                                      intel-ai-safety-1.0.0-explainers           d8219a0cf128   About an hour ago    3.24GB
```

## Model Card Generator

### Build the image
```bash
docker compose build model_card_gen
```

### Check existing image
```bash
docker images | grep -i mcg
intel/ai-tools                                      intel-ai-safety-1.0.0-mcg                  82bdf7b239cc   About a minute ago   3.02GB
```
## Model Card Generator UI

### Build the image
```bash
docker compose build model_card_gen_ui
```

### Check existing image
```bash
docker images | grep -i mcg-ui
intel/ai-tools                                      intel-ai-safety-1.0.0-mcg-ui        e9bd59328e37   About a minute ago    2.75GB
```

### Running the UI
To run the Model Card Generator UI, you can use the docker run command. 
```bash
docker run --rm -p 8051:8051 --name mcg-ui intel/ai-tools:intel-ai-safety-1.0.0-mcg-ui
```
Once the container is running, you can access the Model Card Generator UI by navigating to `<HOST_NAME>:8051` in your web browser, where HOST_NAME is the name or IP address of the server that the container is running on.

Docker containers can run in either Interactive or Jupyter mode.

### Interactive
This mode allows running the container in an interactive shell. This enables the ability to interact with the container's bash shell. Below is the command to start the container in interactive mode:

```bash
docker run --rm -it intel/ai-tools:intel-ai-safety-1.0.0-explainers bash
```
or
```bash
docker run --rm -it intel/ai-tools:intel-ai-safety-1.0.0-mcg bash
```

### Jupyter
This mode launches a jupyterlab notebook server. The command below will start the jupyterlab server which can be accessed from a web browser. Each container includes jupyter kernel to enable conda environment in jupyter notebook. The port for this server is `8888` and is exposed by default when you run the container, but you can change to any port that's open on the host. In this example we are using `8887` for `explainers` container and `8889` for `mcg` container on the host.

```bash
docker run --rm -p 8887:8888 --name explainers intel/ai-tools:intel-ai-safety-1.0.0-explainers
```
or
```bash
docker run --rm -p 8889:8888 --name model-card-gen intel/ai-tools:intel-ai-safety-1.0.0-mcg
```

You can also run these containers in daemon mode:
```bash
docker run --rm -d -p 8887:8888 --name explainers intel/ai-tools:intel-ai-safety-1.0.0-explainers
```
or
```bash
docker run --rm -d -p 8889:8888 --name model-card-gen intel/ai-tools:intel-ai-safety-1.0.0-mcg
```

Finally, on your favorite browser, navigate to `<HOST_NAME>:8887` where `HOST_NAME` is the name or IP address of the server that the container is running on. If asked for a token, review the container logs to locate the token for the Jupyter server.

```bash
docker logs -f model-card-gen
```
or
```bash
docker logs -f explainers
```

## Final notes
To run the containers, you can also use `docker compose` this way for example:
```bash
docker compose run model_card_gen
```
or to run in the container in daemon mode:
```bash
docker compose run -d model_card_gen
```

These containers are built with `intelai` as default non-root container user.
If you prefer to run these containers with a different user, you can build them with custom `build-arg`'s.
For example this command builds the containers with current system user along with user's id and group:
```bash
docker compose build --build-arg NON_ROOT_USER=$(id -un) --build-arg UID=(id -u) --build-arg GID=$(id -g)
```
