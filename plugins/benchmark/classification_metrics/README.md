# Toxicity Detection Accuracy

Toxicity detection plays a critical role in guarding the inputs and outputs of large language models (LLMs) to ensure safe, respectful, and responsible content. Given the widespread use of LLMs in applications like customer service, education, and social media, there's a significant risk that they could inadvertently produce or amplify harmful language if toxicity is not detected effectively. 

For evaluating a target toxicity detection LLM, we use the ToxicChat dataset and the most commonly used metrics in toxicity classification, to provide a comprehensive assessment. The Gaudi 2 accelerator is deployed in the benchmark to address the high demand of the AI workload while balancing the power efficiency. 

- Dataset
    - [ToxicChat](https://huggingface.co/datasets/lmsys/toxic-chat)
- Metrics
    - accuracy
    - auroc
    - f1
    - precision
    - recall

## Get Started

### Requirements
```bash
git clone https://github.com/huggingface/optimum-habana.git
```
### Setup
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


Test the model and confirm the results saved correctly

```bash
MODEL_PATH=Intel/toxic-prompt-roberta
DATASETS=tc
python ./scripts/benchmark_accuracy.py -m ${MODEL_PATH} -d ${DATASETS}
cat ${MODEL_PATH%%/*}/results/${MODEL_PATH##*/}_${DATASETS}_accuracy/metrics.json
```