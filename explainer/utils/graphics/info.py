#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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

from dataclasses import dataclass
import ipywidgets as widgets
from IPython.display import display
from typing import List, Tuple


@dataclass
class InfoField:
    name: str
    description: str

@dataclass(init=False)
class InfoPanel:
    """Dataclass to build plot information pannel in Jupyter enviornment
    """
    fields: List[InfoField]
    title = "Metric Info"
        
    def __init__(self, *args: Tuple[str], **kwargs : str):
        """
        Args:
            *args: Variable length argument list of tuples such that the first
                item is a field name and second item is field description.
            *kwargs: Keyword arguments such that the key is a field name and
                value is the field description.

        """
        if args:
            self.fields = [InfoField(*arg) for arg in args]
        elif kwargs:
            self.fields = [InfoField(k, v) for k, v in kwargs.items()]
            
    def show(self):
        """Display widget object to Jupyter environment
        """
        output = self._build_widget()
        display(output)
    
    def _build_widget(self):
        """Build ipywidgets Accordion object with list from data fields 
        """
        output = widgets.Accordion(children=[])
        output.children  = [widgets.HTML((self._build_html(self.fields)))]
        output.set_title(0, self.title)
        return output

    def _build_html(self, elms):
        """Build HTML list content for Jupyter widget
        """
        html_inner=''
        for elm in elms:
            html_inner += "<li><b>{}: </b>{}</li>".format(elm.name, elm.description)
        html_outer = "<ul>{}</ul>".format(html_inner)
        return html_outer
