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
CLI build subcommand

The build subcommand will build a wheel under explainer/plugins/'plugin' using the 
setup.py created by the generate subcommand. It will then move the wheel from 
explainer/plugins/'plugin'/dist to explainer/plugins, so it can be installed
using the install subcommand.

"""
import click
from explainer.cli import (complete_explainers, pass_environment, Environment)


@click.command("build", short_help="builds a wheel and moves it to explainer/plugins")
@click.argument("path", required=True, type=str, shell_complete=complete_explainers)
@pass_environment
def cli(env: Environment, path: str):
    """Passes the path to explainer.build
    """
    env.explainer.build(path)
