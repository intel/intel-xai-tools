"""
Explainer entry point
"""
import os
import sys
import click

CONTEXT_SETTINGS = dict(auto_envvar_prefix="EXPLAINER")

class Environment:
    """
    Provides an Environment that can be passed to click subcommands
    """
    def __init__(self):
        self.verbose = False
        self.home = os.getcwd()

    def log(self, msg, *args):
        """Logs a message to stderr."""
        if args:
            msg %= args
        click.echo(msg, file=sys.stderr)

    def vlog(self, msg, *args):
        """Logs a message to stderr only if verbose is enabled."""
        if self.verbose:
            self.log(msg, *args)

pass_environment = click.make_pass_decorator(Environment, ensure=True)
cmd_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "commands"))

class ComplexCLI(click.MultiCommand):
    """_summary_

    Args:
        click (_type_): _description_
    """
    def list_commands(self, ctx):
        """Return the subcommands under the subdir commands

        Args:
            ctx (_type_): _description_

        Returns:
            _type_: _description_
        """
        r_var = []
        for filename in os.listdir(cmd_folder):
            if filename.endswith(".py") and filename.startswith("cmd_"):
                r_var.append(filename[4:-3])
        r_var.sort()
        return r_var

    def get_command(self, ctx, cmd_name):
        """_summary_

        Args:
            ctx (_type_): _description_
            cmd_name (_type_): _description_

        Returns:
            _type_: _description_
        """
        try:
            mod = __import__(f"explainer.commands.cmd_{cmd_name}", None, None, ["cli"])
        except ImportError:
            return
        return mod.cli


@click.command(cls=ComplexCLI, context_settings=CONTEXT_SETTINGS)
@click.option("-v", "--verbose", is_flag=True, help="Enables verbose mode.")
@pass_environment
def cli(ctx, verbose):
    """explainer CLI."""
    ctx.verbose = verbose
