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
