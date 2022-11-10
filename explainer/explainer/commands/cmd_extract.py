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
CLI extract subcommand

The extract subcommand will create a plugins directory using
information in the explainer/explainers yaml file, which is the first argument
to the command. It will call generate followed by replacing the entry_point python 
file with what is in the associated wheel

"""
import click
from explainer.cli import (complete_explainers, pass_environment, Environment)


@click.command("extract", short_help="unpacks a wheel file to explainer/plugins")
@click.argument("path", required=True, type=str, shell_complete=complete_explainers)
@pass_environment
def cli(env: Environment, path: str):
    """Passes the path to explainer.generate
    """
    env.explainer.extract(path)
