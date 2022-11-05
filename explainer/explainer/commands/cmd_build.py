"""
CLI build subcommand

The build subcommand will build a wheel under explainer/plugins/'plugin' using the 
setup.py created by the generate subcommand. It will then move the wheel from 
explainer/plugins/'plugin'/dist to explainer/plugins, so it can be installed
using the install subcommand.

"""
import click
from explainer.cli import (complete_explainers, pass_environment, Environment)


@click.command("build", short_help="builds a wheel and moves it to explainer/plugins")
@click.argument("path", required=True, type=str, shell_complete=complete_explainers)
@pass_environment
def cli(env: Environment, path: str):
    """Passes the path to explainer.build
    """
    env.explainer.build(path)
