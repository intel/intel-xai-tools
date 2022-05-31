"""CLI entry point for the explain subcommand"""
import pickle
import click
import pandas as pd
import yaml
from explainer.cli import (pass_environment, Environment)


@click.command("explain", short_help="Explains the model and data ")
@click.option('--data', '-d', 'datapath', required=True, help='path to dataset')
@pass_environment
def cli(env: Environment, datapath: str, outputpath: str):
    """calls an explainer providing the model, dataset and features

    Args:
        env (Environment): the Environment class that holds context for this subcommand
        datapath (str): the datapath
        outputpath (str): _description_
    """
    click.echo(f"data is {datapath}")
    _data = pd.read_pickle(datapath)
