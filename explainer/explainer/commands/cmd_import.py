"""
CLI import subcommand

The import subcommand loads an exlainer plugin into the current python environment.
It does this by adding the plugin installation directory to the head of the sys.path
and instantiating an ExplainerModuleSpec that holds the plugins entry_points.
These entry_points can be called with the appropriate arguments.

Explainer plugins provide different entry_points that serve
as ways to access XAI methods bundled within the plugin. 

"""
import click
from typing import List, Union
from importlib.metadata import EntryPoint
from explainer.cli import (complete_explainers, pass_environment, Environment)


@click.command("import",
               short_help="imports an explainable's functionality as defined by ExplainerSpec")
@click.argument("path", required=True, type=str, shell_complete=complete_explainers)
@click.argument("entry_point", required=False, type=str)
@click.argument('args', type=str, nargs=-1)
@pass_environment
def cli(env: Environment, path: str, entry_point: Union[str, None] = None,
        args: Union[List[str], None] = None):
    """Passes the path to explainer.import_from

    """
    modspec = env.explainer.import_from(path)
    if modspec is not None:
        if entry_point is not None:
            ep: Union[EntryPoint, None] = modspec[entry_point]
            if ep is not None:
                if args is not None:
                    ep(' '.join(args))
                else:
                    ep()
