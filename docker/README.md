## Intel® Explainable AI Tools Dockerfiles and images
Currently there are Dockerfiles for both `Model Card Generator` and `Explainers`.

* [Model Card Generator](#model-card-generator)
* [Explainers](#explainers)
* [Interactive](#interactive)
* [Jupyter](#jupyter)

## Model Card Generator

### Build the image
```bash
docker build --pull -f docker/Dockerfile.mcg -t intel-ai-safety:mcg --target runtime .
```

### Check existing images:
```bash
docker images | grep -i mcg
intel-ai-safety                              mcg        e762b7e97e42   5 hours ago    2.99GB
```

## Explainers

### Build the image
```bash
docker build --pull -f docker/Dockerfile.explainers -t intel-ai-safety:explainers --target runtime .
```

### Check existing images:
```bash
docker images | grep -i explainers
intel-ai-safety                              explainers   48c6130b1f02   22 minutes ago   3.23GB
```

Docker containers can run in either Interactive or Jupyter mode.

### Interactive
This mode allows running the container in an interactive shell. This enables the ability to interact with the container's bash shell. Below is the command to start the container in interactive mode:
```bash
docker run --rm -it intel-ai-safety:mcg bash
```
or
```bash
docker run --rm -it intel-ai-safety:explainers bash
```

### Jupyter
This mode launches a jupyterlab notebook server. The command below will start the jupyterlab server which can be accessed from a web browser. Each container includes jupyter kernel to enable conda environment in jupyter notebook. The port for this server is `8888` and is exposed by default when you run the container, but you can change to any port that's open on the host. In this example we are using `9999` on the host.

```bash
docker run --rm -p 9999:8888 --name model-card-gen intel-ai-safety:mcg
```
or
```bash
docker run --rm -p 9999:8888 --name model-card-gen intel-ai-safety:explainers
```
You can also run these containers in daemon mode:
```bash
 docker run --rm -d -p 9999:8888 --name model-card-gen intel-ai-safety:mcg
```
or
```bash
docker run --rm -d -p 9999:8888 --name explainers intel-ai-safety:explainers
```

Finally, on your favorite browser, navigate to `<HOST_NAME>:9999` where `HOST_NAME` is the name or IP address of the server that the container is running on. If asked for a token, review the container logs to locate the token for the Jupyter server.

```bash
docker logs -f model-card-gen
```
or
```bash
docker logs -f explainers
```
