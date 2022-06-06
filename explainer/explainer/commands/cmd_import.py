"""CLI entry point for the list subcommand"""
import click
from explainer.cli import (pass_environment, Environment)


@click.command("import", short_help="imports an explainable's functionality as defined by ExplainerSpec")
@pass_environment
def cli(_env: Environment):
    """Processes the model and data and return values to stdout

    """
    click.echo("list")
