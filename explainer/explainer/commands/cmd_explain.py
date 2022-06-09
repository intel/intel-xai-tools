"""CLI entry point for the explain subcommand"""
import click
from explainer.cli import (pass_environment, Environment)


@click.command("explain", short_help="Calls the ModuleSpec entry point")
@pass_environment
def cli(_env: Environment):
    """calls an explainer providing the model, dataset and features

    Args:
        env (Environment): the Environment class that holds context for this subcommand
        datapath (str): the datapath
        outputpath (str): _description_
    """
    click.echo("explain method")
