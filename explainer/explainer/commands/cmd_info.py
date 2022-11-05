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
        print(f"{spec!r}")
#        if hasattr(spec, "dependencies"):
#            if spec.dependencies is not None:
#                for dependency in spec.dependencies:
#                    parts: List[str] = dependency.split("==")
#                    package = parts[0]
#                    # drop any [subpackage]
#                    package = package.split("[")[0]
#                    version = parts[1]
#                    url = "https://pypi.org/pypi/" + package + "/" + version + "/json"
#                    response = http.request("GET", url)
#                    if response.status == 200:
#                        package_json = json.loads(response.data)
#                        top_dependencies = package_json['info']['requires_dist']
#                        pprint(f"{package} requires {top_dependencies}")
