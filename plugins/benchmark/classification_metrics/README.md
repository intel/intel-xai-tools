# Toxicity Detection Accuracy

Toxicity detection plays a critical role in guarding the inputs and outputs of large language models (LLMs) to ensure safe, respectful, and responsible content. Given the widespread use of LLMs in applications like customer service, education, and social media, there's a significant risk that they could inadvertently produce or amplify harmful language if toxicity is not detected effectively. 

For evaluating a target toxicity detection LLM, we use the ToxicChat and Jigsaw datasets, and the most commonly used metrics in toxicity classification, to provide a comprehensive assessment. Currently, the benchmark script only supports benchmarking one dataset at a time. Future work includes enabling benchmarking on multiple datasets at a time. The Gaudi 2 accelerator is deployed in the benchmark to address the high demand of the AI workload while balancing the power efficiency. 

- Supported Dataset
    - [ToxicChat](https://huggingface.co/datasets/lmsys/toxic-chat)
    - [Jigsaw Unintended Bias](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)
    - More datasets to come...
    
- Supported Metrics
    - accuracy
    - auprc (area under precision recall curve)
    - auroc
    - f1
    - fpr (false positive rate)
    - precision
    - recall

## Get Started

### Requirements
If you are using an `hpu` device, then clone the `optimum-habana` repository.
```bash
git clone https://github.com/huggingface/optimum-habana.git
```

### Setup
If you're running behind corporate proxy, run Gaudi Docker with additional proxies and volume mount.
```bash
DOCKER_RUN_ENVS="--env ftp_proxy=${ftp_proxy} --env FTP_PROXY=${FTP_PROXY} --env http_proxy=${http_proxy} --env HTTP_PROXY=${HTTP_PROXY} --env https_proxy=${https_proxy} --env HTTPS_PROXY=${HTTPS_PROXY} --env no_proxy=${no_proxy} --env NO_PROXY=${NO_PROXY} --env socks_proxy=${socks_proxy} --env SOCKS_PROXY=${SOCKS_PROXY} --env TF_ENABLE_MKL_NATIVE_FORMAT=1"

docker run --disable-content-trust ${DOCKER_RUN_ENVS} \
    -d --rm -it --name toxicity-detection-benchmark \
    -v ${PWD}:/workdir \
    --runtime=habana \
    -e HABANA_VISIBLE_DEVICES=all \
    -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
    --cap-add=sys_nice \
    --net=host \
    --ipc=host \
    vault.habana.ai/gaudi-docker/1.16.2/ubuntu22.04/habanalabs/pytorch-installer-2.2.2:latest
```

### Evaluation
#### Execute interactive container
```bash
docker exec -it toxicity-detection-benchmark bash
```
#### Navigate to `workdir` and install required packages
```bash
cd /workdir
cd optimum-habana && pip install . && cd ..
pip install -r requirements.txt
```

In case of [Jigsaw Unintended Bias](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification), make sure the datasets are downloaded and stored in current working directory.

#### Test the model and confirm the results saved correctly
Replace `MODEL_PATH` and `DATASETS` with the appropriate path for the model and the name of the dataset.
```bash
MODEL_PATH=Intel/toxic-prompt-roberta
DATASETS=tc
python ./scripts/benchmark_classification_metrics.py -m ${MODEL_PATH} -d ${DATASETS}
cat ${MODEL_PATH%%/*}/results/${MODEL_PATH##*/}_${DATASETS}_accuracy/metrics.json
```

If you are using an `hpu` device, you can instantiate the Gaudi configuration by passing the `GAUDI_CONFIG_NAME` variable with the appropriate configuration name. The default value for the device name (`device`) is `hpu`.
```bash
MODEL_PATH=Intel/toxic-prompt-roberta
DATASETS=tc
GAUDI_CONFIG_NAME=Habana/roberta-base
DEVICE_NAME=hpu
python ./scripts/benchmark_classification_metrics.py -m ${MODEL_PATH} -d ${DATASETS} -g_config ${GAUDI_CONFIG_NAME} --device ${DEVICE_NAME}
cat ${MODEL_PATH%%/*}/results/${MODEL_PATH##*/}_${DATASETS}_accuracy/metrics.json 
```

For the Jigsaw Unintended Bias, pass the path of the stored dataset file in place of the variable `DATASET_PATH`.
```bash
MODEL_PATH=Intel/toxic-prompt-roberta
DATASETS=jigsaw
DATASET_PATH=/path/to/dataset
python ./scripts/benchmark_classification_metrics.py -m ${MODEL_PATH} -d ${DATASETS} -p ${DATASET_PATH} 
cat ${MODEL_PATH%%/*}/results/${MODEL_PATH##*/}_${DATASETS}_accuracy/metrics.json
```