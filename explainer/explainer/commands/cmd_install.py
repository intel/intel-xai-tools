"""
CLI install subcommand

The install subcommand will install a wheel in a directory under explainer/plugins using the 
information in the explainer/explainers yaml file, which is the first argument
to the command. The yaml file has an attribute `plugin` which names the plugin wheel.
The install command calls `pip install` providing the wheel and a target directory 
which is derived from the path of the yaml file. The target directory path is located 
under explainer/explainers.

"""
import click
from explainer.cli import (complete_explainers, pass_environment, Environment)

@click.command("install", short_help="installs the plugin (wheel) under the directory explainer/explainers")
@click.argument("path", required=True, type=str, shell_complete=complete_explainers)
@pass_environment
def cli(env: Environment, path: str):
    """Passes the path to explainer.export_to
    """
    env.explainer.install(path)
