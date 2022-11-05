"""
CLI uninstall subcommand

The uninstall subcommand will remove the plugin directory under explainer/explainers.
The related yaml file is not touched.

"""
import sys
import click
from explainer.cli import (complete_explainers, pass_environment, Environment)


@click.command("uninstall", short_help="uninstalls the plugin directory under the directory explainer/explainers.")
@click.argument("path", required=True, type=str, shell_complete=complete_explainers)
@pass_environment
def cli(env: Environment, path: str):
    """Passes the path to explainer.uninstall
    """
    env.explainer.uninstall(path)
