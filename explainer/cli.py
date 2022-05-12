"""
Explainer entry point
"""
import os
import sys
from typing import Any
import click
from explainer.api import Explainer

CONTEXT_SETTINGS = dict(auto_envvar_prefix="EXPLAINER")


class Environment:
    """
    Provides an Environment that can be passed to click subcommands
    """

    def __init__(self):
        self.verbose = False
        self.home = os.getcwd()
        self.explainer = Explainer()

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
    """top entry point in click, context is passed down to subcommands

    Args:
        click (_type_): _description_
    """

    def list_commands(self, ctx: Environment) -> list:
        """Return the subcommands under the subdir commands

        Args:
            ctx (click.Context): dict that can be handed to subcommands

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
            _env (Environment): _description_
            cmd_name (str): _description_

        Returns:
            Any: _description_
        """
        try:
            mod = __import__(
                f"explainer.commands.cmd_{cmd_name}", None, None, ["cli"])
        except ImportError:
            return
        return mod.cli


@click.command(cls=ExplainerCLI, context_settings=CONTEXT_SETTINGS)
@pass_environment
def cli(_env: Environment):
    """explainer CLI."""
