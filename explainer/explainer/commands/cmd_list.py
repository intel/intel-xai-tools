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
CLI list subcommand

The list subcommand lists the yaml file basenames under explainer/explainers.
The yaml basename is used by the generate, install and import subcommands.
The yaml file is where one defines the entry_point functions that are available under the plugin.

"""
import click
from explainer.cli import (pass_environment, Environment)


@click.command("list", short_help="lists available explainers")
@pass_environment
def cli(env: Environment):
    """Prints out the available explainers

    """
    for explainer in env.explainer.list:
        click.echo(f"{explainer}")

