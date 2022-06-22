"""CLI entry point for the list subcommand"""
import click
from explainer.cli import (pass_environment, Environment)


@click.command("list", short_help="lists available explainers")
@pass_environment
def cli(env: Environment):
    """Prints out env.explainer.explainables

    """
    for explainable in env.explainer.explainables:
        click.echo(f"{explainable}")

