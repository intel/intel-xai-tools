"""
Explainer CLI
"""
import os
import sys
from importlib import import_module
from typing import Any, List
import click
import click_completion
from explainer.api import Explainer

CONTEXT_SETTINGS = dict(auto_envvar_prefix="EXPLAINER")

click_completion.init()


def complete_explainers(ctx, param, incomplete) -> List[str]:
    exp = Explainer()
    if incomplete is not None:
        return [e for e in exp.list if e.startswith(incomplete)]
    return exp.list


class Environment:
    """ Provides an Environment that can be passed to click subcommands"""

    def __init__(self):
        self.verbose: bool = False
        self.home: str = os.getcwd()
        self.explainer: Explainer = Explainer()

    def log(self, msg: str, *args):
        """Logs a message to stderr."""
        if args:
            msg %= args
        click.echo(msg, file=sys.stderr)

    def vlog(self, msg: str, *args):
        """Logs a message to stderr only if verbose is enabled."""
        if self.verbose:
            self.log(msg, *args)


pass_environment = click.make_pass_decorator(Environment, ensure=True)
cmd_folder = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "commands"))


class ExplainerCLI(click.MultiCommand):
    """ExplainerCLI inherits from click.MultiCommand


    ExplainerCLI will import explainable resources as defined by a yaml file.
    The yaml file has fields for model, data, python resources,
    and a Callable entry point
    """

    def list_commands(self, _env: Environment) -> list:
        """Return the subcommands under the subdir commands/

        Args:
            _env (Environment): environment that holds context

        Returns:
            list: list of commands
        """
        r_var: list = []
        for filename in os.listdir(cmd_folder):
            if filename.endswith(".py") and filename.startswith("cmd_"):
                r_var.append(filename[4:-3])
        r_var.sort()
        return r_var

    def get_command(self, _env: Environment, cmd_name: str) -> Any:
        """dynamically loads command if found under commands subdir

        Args:
            _env (Environment): environment that holds context
            cmd_name (str): the name of the command to load

        Returns:
            Any: cli
        """
        try:
            mod = import_module(f"explainer.commands.cmd_{cmd_name}", "cli")
        except ImportError:
            return
        return mod.cli


@click.group()
def explainer():
    """The explainer command group."""


@explainer.command(cls=ExplainerCLI, context_settings=CONTEXT_SETTINGS)
@pass_environment
def cli(_env: Environment):
    """The explainer CLI enables a plugin model for XAI. Plugins are a recognized way to enable functional categories. See https://packaging.python.org/en/latest/specifications/entry-points/."""
