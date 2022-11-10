#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
#

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
