## 1. Build docker image
```
docker build -t helm-image --file Dockerfile.helm .
```
## 2. Run docker container
```
mkdir -p benchmark_output
docker run \
 -v $(pwd)/benchmark_output:/app/benchmark_output \
 -v $(pwd)/prod_env:/app/helm/prod_env \
 -p 8000:8000 \
 --name helm-container \
 -it helm-image bash
```
### (Optional) Run docker container with your HuggingFace cache
Instead, if you have already downloaded the model(s) you plan to benchmark from HuggingFace hub, mount your HuggingFace cache to skip re-downloading the models(s) by running the following `docker run`:
```
mkdir -p benchmark_output
docker run \
 -v $(pwd)/benchmark_output:/app/benchmark_output \
 -v $(pwd)/prod_env:/app/helm/prod_env \
 -v ~/.cache/huggingface:/root/.cache/huggingface \
 -p 8000:8000 \
 --name helm-container \
 -it helm-image bash
```
## 3. Run HELM
### Option A: Run simple spec config file that runs only one HELM scenario
This is a quick preliminary run to confirm that your environment is working properly.
```
helm-run --conf-paths run_spec_simple.conf --suite simple --max-eval-instances 1
```
If the `helm-run` was successful then the `benchmark_output` directory inside and **outside of your container** should contain the resulting log files as seen below:
```
ls benchmark_output/
runs  scenario_instances  scenarios
```
### Option B: Run a longer spec config file that contains the standard suite of scenarios of HELM-lite
This is a longer run if you have confirmed your environment has been setup properly.
```
helm-run --conf-paths run_specs.conf --suite v1 --max-eval-instances 10
```
### Option C: Run toxic spec config file that contains Real Toxic Prompts
This requires uncommenting and manual entering your Perspective API Key in helm/prod_env. Then, an additional flag is needed to point to the credentials, seen below.
```
helm-run --conf-paths run_spec_toxic.conf --suite v1 --max-eval-instances 10 --local-path helm/credentials.conf
```
## 4. Summarize HELM results
```
helm-summarize --suite v1
```
## 5. Start a web server to display the HELM results
```
helm-server
```
