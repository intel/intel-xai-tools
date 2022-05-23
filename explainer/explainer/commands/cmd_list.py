"""CLI entry point for the list subcommand"""
import click
from explainer.cli import (pass_environment, Environment)


@click.command("list", short_help="lists available explainers")
@pass_environment
def cli(_env: Environment):
    """Processes the model and data and return values to stdout

    """
    click.echo("list")
