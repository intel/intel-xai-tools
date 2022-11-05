"""
CLI list subcommand

The list subcommand lists the yaml file basenames under explainer/explainers.
The yaml basename is used by the generate, install and import subcommands.
The yaml file is where one defines the entry_point functions that are available under the plugin.

"""
import click
from explainer.cli import (pass_environment, Environment)


@click.command("list", short_help="lists available explainers")
@pass_environment
def cli(env: Environment):
    """Prints out the available explainers

    """
    for explainer in env.explainer.list:
        click.echo(f"{explainer}")

