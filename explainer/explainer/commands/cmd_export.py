"""CLI entry point for the export subcommand"""
import click
from explainer.cli import (pass_environment, Environment)


@click.command("export", short_help="exports packaages to a zip file")
@click.argument("path", required=True, type=click.Path(resolve_path=True))
@pass_environment
def cli(env: Environment, path):
    """Passes the path to explainer.export_to
    """
    env.explainer.export_to(path)
