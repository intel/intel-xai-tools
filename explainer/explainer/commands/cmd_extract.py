"""
CLI extract subcommand

The extract subcommand will create a plugins directory using
information in the explainer/explainers yaml file, which is the first argument
to the command. It will call generate followed by replacing the entry_point python 
file with what is in the associated wheel

"""
import click
from explainer.cli import (complete_explainers, pass_environment, Environment)


@click.command("extract", short_help="unpacks a wheel file to explainer/plugins")
@click.argument("path", required=True, type=str, shell_complete=complete_explainers)
@pass_environment
def cli(env: Environment, path: str):
    """Passes the path to explainer.generate
    """
    env.explainer.extract(path)
