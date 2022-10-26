"""
CLI create subcommand

The create subcommand creates a yaml file under explainer/exlainers using the first argument
to the command which is base name of the yaml. The create command will add
the yaml file's attributes (see below), which then are filled in by the user.
The name used to create the yaml file should not clash with any top-level package names.

.. code-block:: yaml

   --- !ExplainerSpec
   name: <name>
   version: 0.1
   plugin: explainer_explainers_<name>-<version>-py2.py3-none-any.whl
   dependencies: []
   entry_points: []

"""
import click
from explainer.cli import (complete_explainers, pass_environment, Environment)


@click.command("create", short_help="creates a yaml file under explainer/explainers")
@click.argument("yamlname", required=True, type=str, shell_complete=complete_explainers)
@pass_environment
def cli(env: Environment, yamlname: str):
    """Passes the yamlname to explainer.create
    """
    env.explainer.create(yamlname)
