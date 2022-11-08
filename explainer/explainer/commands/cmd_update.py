"""
CLI update subcommand

The update subcommand calls the Explainer update API, which then calls the generate, build and install apis.

"""
import click
from explainer.cli import (complete_explainers, pass_environment, Environment)


@click.command("update", short_help="calls the generate, build and install apis")
@click.argument("yamlname", required=True, type=str, shell_complete=complete_explainers)
@pass_environment
def cli(env: Environment, yamlname: str):
    """Calls explainer(yaml) to set the yamlname and then calls explainer.update.
    """
    env.explainer(yamlname).update
