---
license: apache-2.0
title: Model Card Generator
sdk: streamlit
emoji: 🚀
colorFrom: yellow
colorTo: blue
app_file: home.py
short_description: Create interactive Model Card
---
# Model Card Generator UI

This streamlit application allows users to create interactive HTML reports containing model performance and fairness metrics using a simple interface. Users can either create a new Model Card or modify an existing one.

## Install
1. Choose a virtual enviornment to use: eg. Using virtualenv:

```bash
python3 -m virtualenv mgc_ui_venv
source mgc_ui_venv/bin/activate
```

2. To install the required Python packages, run the following command in your terminal:

```bash
pip install -r requirements.txt
```

## Run the Streamlit Application
You can run the Streamlit application using the following command:

```bash
streamlit run home.py
```

The command will start the Streamlit server, and you should see output in the terminal that includes a URL where you can view the application. Open the URL in your web browser to interact with the Streamlit UI.
