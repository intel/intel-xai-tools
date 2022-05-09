"""CLI entry point for the explainer command"""
import pickle
import click
import pandas as pd
import shap
import yaml


@click.command("process", short_help="Processes the model and data ")
@click.option('--model', '-m', 'modelpath', required=True, help='path to model')
@click.option('--data', '-d', 'datapath', required=True, help='path to dataset')
@click.option('--output', "-o", 'outputpath', required=False, help='output is an explanation json object')
def cli(modelpath, datapath, outputpath):
    click.echo(f"model is {modelpath} data is {datapath}")
    model = pickle.load(open(modelpath, "rb"))
    data = pd.read_pickle(datapath)
    shap_values = shap.TreeExplainer(model).shap_values(data)
    shap_values_json = dumps(shap_values)
    if(outputpath is not None):
        pickle.dump(shap_values_json, open(outputpath, "wb"))
