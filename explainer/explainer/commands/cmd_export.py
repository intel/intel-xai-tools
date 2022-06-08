"""CLI entry point for the export subcommand"""
import click
from explainer.cli import (pass_environment, Environment)


@click.command("export", short_help="exports packaages to a zip file")
@pass_environment
def cli(_env: Environment):
    """Processes the model and data and return values to stdout

    """
    click.echo("export")
