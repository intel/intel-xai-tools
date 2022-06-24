"""CLI entry point for the list subcommand"""
import click
from explainer.cli import (pass_environment, Environment)


@click.command("import",
               short_help="imports an explainable's functionality as defined by ExplainerSpec")
@click.argument("path", required=True, type=click.Path(resolve_path=True))
@pass_environment
def cli(env: Environment, path: str):
    """Passes the path to explainer.import_from

    """
    env.explainer.import_from(path)
