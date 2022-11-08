"""
CLI info subcommand

The info subcommand prints out details of the yaml file.

"""
import os
import click
import certifi
import urllib3
import json
from urllib3 import ProxyManager
from typing import List
from explainer.cli import (pass_environment, complete_explainers, Environment)
from pprint import pprint


@click.command("info",
               short_help="shows info about an explainer under explainer.explainers")
@click.argument("path", required=True, type=str, shell_complete=complete_explainers)
@pass_environment
def cli(env: Environment, path: str):
    """Passes the path to explainer.info

    """
    if 'http' not in locals().keys():
        http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED', ca_certs=certifi.where())
        if 'https_proxy' in os.environ:
            proxy = os.environ['https_proxy']
            http = ProxyManager(proxy)
    module = env.explainer.info(path)
    if hasattr(module, "spec"):
        spec = module.spec
        pprint(spec)
