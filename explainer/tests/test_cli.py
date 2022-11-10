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

import pytest
import sys
from click.testing import CliRunner
from explainer import cli


@pytest.fixture
def runner():
    return CliRunner()

def test_cli_usage(runner):
    result = runner.invoke(cli.cli)
    assert result.exit_code == 0
    assert not result.exception
    usage = "Usage: cli [OPTIONS] COMMAND [ARGS]...\n\n  The explainer CLI enables a plugin model for XAI. Plugins are a recognized way\n  to enable functional categories. See\n  https://packaging.python.org/en/latest/specifications/entry-points/.\n\nOptions:\n  --help  Show this message and exit.\n\nCommands:\n  build      builds a wheel and moves it to explainer/plugins\n  create     creates a yaml file under explainer/explainers\n  extract    unpacks a wheel file to explainer/plugins\n  generate   generates a template plugin directory under explainer/plugins\n  import     imports an explainable's functionality as defined by ExplainerSpec\n  info       shows info about an explainer under explainer.explainers\n  install    installs the plugin (wheel) under the directory\n             explainer/explainers\n  list       lists available explainers\n  uninstall  uninstalls the plugin directory under the directory\n             explainer/explainers.\n  update     Calls the generate, build and install apis\n".split('\n')
    output = result.output.split('\n')
    for index, line in enumerate(output):
        assert line == usage[index]

def test_cli_build_usage_subcommand(runner):
    result = runner.invoke(cli.cli, ['build'])
    assert result.exit_code == 2
    assert result.exception
    usage = "Usage: cli build [OPTIONS] PATH\nTry 'cli build --help' for help.\n\nError: Missing argument 'PATH'.\n"
    usage = usage.split('\n')
    output = result.output.split('\n')
    for index, line in enumerate(output):
        assert line == usage[index]

def test_cli_create_usage_subcommand(runner):
    result = runner.invoke(cli.cli, ['create'])
    assert result.exit_code == 2
    assert result.exception
    usage = "Usage: cli create [OPTIONS] YAMLNAME\nTry 'cli create --help' for help.\n\nError: Missing argument 'YAMLNAME'.\n"
    usage = usage.split('\n')
    output = result.output.split('\n')
    for index, line in enumerate(output):
        assert line == usage[index]

def test_cli_extract_usage_subcommand(runner):
    result = runner.invoke(cli.cli, ['extract'])
    assert result.exit_code == 2
    assert result.exception
    usage = "Usage: cli extract [OPTIONS] PATH\nTry 'cli extract --help' for help.\n\nError: Missing argument 'PATH'.\n"
    usage = usage.split('\n')
    output = result.output.split('\n')
    for index, line in enumerate(output):
        assert line == usage[index]

def test_cli_generate_usage_subcommand(runner):
    result = runner.invoke(cli.cli, ['generate'])
    assert result.exit_code == 2
    assert result.exception
    usage = "Usage: cli generate [OPTIONS] PATH\nTry 'cli generate --help' for help.\n\nError: Missing argument 'PATH'.\n"
    usage = usage.split('\n')
    output = result.output.split('\n')
    for index, line in enumerate(output):
        assert line == usage[index]

def test_cli_import_usage_subcommand(runner):
    result = runner.invoke(cli.cli, ['import'])
    assert result.exit_code == 2
    assert result.exception
    usage = "Usage: cli import [OPTIONS] PATH [ENTRY_POINT] [ARGS]...\nTry 'cli import --help' for help.\n\nError: Missing argument 'PATH'.\n"
    usage = usage.split('\n')
    output = result.output.split('\n')
    for index, line in enumerate(output):
        assert line == usage[index]

def test_cli_info_usage_subcommand(runner):
    result = runner.invoke(cli.cli, ['info'])
    assert result.exit_code == 2
    assert result.exception
    usage = "Usage: cli info [OPTIONS] PATH\nTry 'cli info --help' for help.\n\nError: Missing argument 'PATH'.\n"
    usage = usage.split('\n')
    output = result.output.split('\n')
    for index, line in enumerate(output):
        assert line == usage[index]

def test_cli_install_usage_subcommand(runner):
    result = runner.invoke(cli.cli, ['install'])
    assert result.exit_code == 2
    assert result.exception
    usage = "Usage: cli install [OPTIONS] PATH\nTry 'cli install --help' for help.\n\nError: Missing argument 'PATH'.\n"
    usage = usage.split('\n')
    output = result.output.split('\n')
    for index, line in enumerate(output):
        assert line == usage[index]

def test_cli_list_subcommand(runner):
    result = runner.invoke(cli.cli, ['list'])
    assert result.exit_code == 0
    assert not result.exception
    usage = 'dashboard_explainer\nfeature_attributions_explainer\nlm_classifier_explainer\nlm_layers_explainer\nlm_zeroshot_explainer\nmetrics_explainer\nsetfit_explainer\ntest_explainer\nzero_shot_learning\n'
    usage = usage.split('\n')
    output = result.output.split('\n')
    for index, line in enumerate(output):
        assert line == usage[index]

def test_cli_uninstall_usage_subcommand(runner):
    result = runner.invoke(cli.cli, ['uninstall'])
    assert result.exit_code == 2
    assert result.exception
    usage = "Usage: cli uninstall [OPTIONS] PATH\nTry 'cli uninstall --help' for help.\n\nError: Missing argument 'PATH'.\n"
    usage = usage.split('\n')
    output = result.output.split('\n')
    for index, line in enumerate(output):
        assert line == usage[index]

def test_cli_update_usage_subcommand(runner):
    result = runner.invoke(cli.cli, ['update'])
    assert result.exit_code == 2
    assert result.exception
    usage = "Usage: cli update [OPTIONS] YAMLNAME\nTry 'cli update --help' for help.\n\nError: Missing argument 'YAMLNAME'.\n"
    usage = usage.split('\n')
    output = result.output.split('\n')
    for index, line in enumerate(output):
        assert line == usage[index]

#def test_cli_with_option(runner):
#    result = runner.invoke(cli.cli, ['--as-cowboy'])
#    assert not result.exception
#    assert result.exit_code == 0

#def test_cli_build_2_subcommand(runner):
#    result = runner.invoke(cli.cli, ['build'])
#    assert result.exit_code == 2
#    assert not result.exception
#    breakpoint()
#    assert result.output.strip() == 'Hello, Kam.'
