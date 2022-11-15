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
CLI update subcommand

The update subcommand calls the Explainer update API, which then calls the generate, build and install apis.

"""
import click
from explainer.cli import (complete_explainers, pass_environment, Environment)


@click.command("update", short_help="calls the generate, build and install apis")
@click.argument("yamlname", required=True, type=str, shell_complete=complete_explainers)
@pass_environment
def cli(env: Environment, yamlname: str):
    """Calls explainer(yaml) to set the yamlname and then calls explainer.update.
    """
    env.explainer(yamlname).update
