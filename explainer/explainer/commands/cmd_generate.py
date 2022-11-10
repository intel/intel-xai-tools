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
CLI generate subcommand

The generate subcommand creates a directory under explainer/plugins using the 
information in the explainer/explainers yaml file, which is the first argument
to the command. The base name of the yaml file will be used to create the directory. 
The generate command generates a set of files which can be used to develop the plugin.
The set of files created by generate are:

- ✅ '<plugin>'/README.md
- ✅ '<plugin>'/'basename yaml file'
- ✅ '<plugin>'/Makefile
- ✅ '<plugin>'/setup.cfg
- ✅ '<plugin>'/setup.py
- '<plugin>'/test/test.py

The python file created is where the entry_point functions specified in the yaml file are defined.

"""
import click
from explainer.cli import (complete_explainers, pass_environment, Environment)


@click.command("generate", short_help="generates a template plugin directory under explainer/plugins")
@click.argument("path", required=True, type=str, shell_complete=complete_explainers)
@pass_environment
def cli(env: Environment, path: str):
    """Passes the path to explainer.generate
    """
    env.explainer.generate(path)
