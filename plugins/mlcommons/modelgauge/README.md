# Modelgauge Neural-Chat Plugin

## Get Started

### Requirements
* Linux system or WSL2 on Windows (validated on Ubuntu* 20.04/22.04 LTS)
* Python 3.10
* Poetry

## Install
1. Choose a virtual enviornment to use: eg. Using virtualenv:

```bash
python3 -m virtualenv mg_env
source mg_env/bin/activate
```

2. Navigate to `modelgauge/suts`. In `neuralchat_sut.py`, modify the `HUGGING_FACE_REPO_ID` and `UNIQUE_ID` variables. Set `HUGGING_FACE_REPO_ID` to correspond with the repository ID of the model you intend to test. Assign a unique identifier to the `UNIQUE_ID` variable for the SUT:

```bash
cd modelgauge/suts
```

3.  Resolve and lock the dependencies:

```bash
poetry lock
```

4.  Install dependencies, and register System Under Test (SUTs) and tests:

 ```bash
poetry install
```

## Run Tests
1. Verify the registration of the SUT or the specific test you wish to execute:

```bash
modelgauge list
```
To view a list of only the SUTs or the tests, you can use the corresponding commands. For a comprehensive list of available commands run the following command:

```bash
 modelgauge
```

2. Enter your keys into the `secrets.toml` file found in the `config` folder.

3. To illustrate, the following command executes the `bbq` test on the `Intel/neural-chat-7b-v3-3` SUT:

```shell
modelgauge run-test --sut Intel/neural-chat-7b-v3-3 --test bbq
```
The results from the test will be saved in the output directory.
