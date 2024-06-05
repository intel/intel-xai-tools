#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Intel Corporation
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
from pydoc import locate


def get_plugin_extended_cls(pkg_path, namespace="intel_ai_safety"):
    """Retrieve class from plugin otherwise raise error if pluign not found."""
    pkg_path = f"{namespace}.{pkg_path}"
    plugin_extended_cls = locate(pkg_path)
    if plugin_extended_cls is None:
        raise ModuleNotFoundError(f"Please install {pkg_path} plugin.")
    return plugin_extended_cls
