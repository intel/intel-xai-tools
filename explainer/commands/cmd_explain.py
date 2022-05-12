"""CLI entry point for the explain subcommand"""
import pickle
import click
import pandas as pd
import shap
import yaml
from explainer.cli import (pass_environment, Environment)


@click.command("explain", short_help="Explains the model and data ")
@click.option('--model', '-m', 'modelpath', required=True, help='path to model')
@click.option('--data', '-d', 'datapath', required=True, help='path to dataset')
@click.option('--output', "-o", 'outputpath', required=False, help='explanation json object')
@pass_environment
def cli(_env: Environment, modelpath: str, datapath: str, outputpath: str):
    """calls an explainer providing the model, dataset and feature

    Args:
        env (Environment): the Environment class that holds context for this subcommand
        modelpath (str): _description_
        datapath (str): _description_
        outputpath (str): _description_
    """
    click.echo(f"model is {modelpath} data is {datapath}")
    model = pickle.load(open(modelpath, "rb"))
    data = pd.read_pickle(datapath)
    shap_values = shap.TreeExplainer(model).shap_values(data)
    shap_values_json = yaml.dump(shap_values)
    if outputpath is not None:
        pickle.dump(shap_values_json, open(outputpath, "wb"))
