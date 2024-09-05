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

"""
This module allows users to classes for performing end to end workflows
around model card generation.
"""

# Base
import os
import pkgutil
import tempfile

# External
import jinja2
import pandas as pd
from IPython.display import display, HTML, Markdown

# Internal
from intel_ai_safety.model_card_gen.model_card import ModelCard
from intel_ai_safety.model_card_gen.graphics.add_graphics import (
    add_dataset_feature_statistics_plots,
    add_overview_graphs,
    add_eval_result_plots,
    add_eval_result_slicing_metrics

)

# Typing
from typing import Optional, Sequence, Text, Union, Dict, Any

DataFormat = Union[pd.DataFrame, Text]

_UI_TEMPLATES = {
    "md":(
    "template/md/default_template.md.jinja",
    "template/html/style/default_style.html.jinja",
    "template/md/macros/default_macros.md.jinja"),
    "html":(
    "template/html/default_template.html.jinja",
    "template/html/style/default_style.html.jinja",
    "template/html/macros/default_macros.html.jinja",
    "template/html/js/plotly_js_header.html.jinja",)
    }

_MC_JSON_FILE = os.path.join("data", "model_card.json")
_TEMPLATE_DIR = "template"
_MODEL_CARDS_DIR = "model_cards"
_DEFAULT_UI_TEMPLATE_FILE = {
    "md":os.path.join("md", "default_template.md.jinja"),
    "html":os.path.join("html", "default_template.html.jinja")
}

class ModelCardGen:
    """Generate ModelCard from with TFMA

    Args:
        model_card (ModelCard or dict): pre-generated ModelCard Python object or dictionary following model card schema
        output_dir (str): representing of where to output model card
    """

    def __init__(
        self,
        model_card: Union[ModelCard, Dict[Text, Any]] = None,
        metrics_by_threshold: Union[pd.DataFrame, Text]= None,
        metrics_by_group: Union[pd.DataFrame, Text] = None,
        data_sets: Dict[Text, DataFormat] = {},
        data_stats: Dict[Text, pd.DataFrame] = {},
        output_dir: Text = "",
        template_type: Text = "html"
    ):
        self.data_sets = data_sets
        self.data_stats = data_stats
        self.template_type = template_type
        
        if isinstance(metrics_by_threshold, pd.DataFrame):
            self.metrics_by_threshold = metrics_by_threshold
        elif isinstance(metrics_by_threshold, str) and os.path.exists(metrics_by_threshold):
            self.metrics_by_threshold = pd.read_csv(metrics_by_threshold)
        else:
            self.metrics_by_threshold = None

        if isinstance(metrics_by_group, pd.DataFrame):
            self.metrics_by_group = metrics_by_group
        elif isinstance(metrics_by_group, str) and os.path.exists(metrics_by_group):
            self.metrics_by_group = pd.read_csv(metrics_by_group)
        else:
            self.metrics_by_group = None
        
        # Local asset paths
        self.output_dir = output_dir or tempfile.mkdtemp()
        self._mc_json_file = os.path.join(self.output_dir, _MC_JSON_FILE)
        self._mc_template_dir = os.path.join(self.output_dir, _TEMPLATE_DIR)
        self._model_cards_dir = os.path.join(self.output_dir, _MODEL_CARDS_DIR)
        # Construct ModelCard object 
        if isinstance(model_card, ModelCard):
            self.model_card = model_card
        elif isinstance(model_card, dict) or isinstance(model_card, str):
            self.model_card = ModelCardGen._read_json(model_card)
        else:
            self.model_card = ModelCard()
        # Generated Attributes
        self.model_card_html = ""
        self.model_card_md = ""

    @classmethod
    def generate(
        cls,
        model_card: Union[ModelCard, Dict[Text, Any], Text] = None,
        metrics_by_threshold: pd.DataFrame= None,
        metrics_by_group: pd.DataFrame = None,
        output_dir: Text = "",
        template_type: Text = "html" 
    ):
        """Class Factory starting TFMA analysis and generating ModelCard

        Args:
            model_card (ModelCard or dict): pre-generated ModelCard Python object or dictionary following model card schema
            output_dir (str): representing of where to output model card
            template_type (str): type of template (html for html templates and md for markdown templates) to use for rendering the model card. Defaults to "html".

        Returns:
            ModelCardGen

        Example:
            >>> from model_card_gen.model_card_gen import ModelCardGen
            >>> model_path = 'compas/model'
            >>> data_paths = {'eval': 'compas/eval.tfrecord', 'train': 'compas/train.tfrecord'}
            >>> eval_config = 'compas/eval_config.proto'
            >>> mcg = ModelCardGen.generate(data_paths, model_path, eval_config, template_type='html') #doctest:+SKIP
        """
        self = cls(
            model_card=model_card,
            output_dir=output_dir,
            metrics_by_threshold=metrics_by_threshold,
            metrics_by_group=metrics_by_group,
            template_type=template_type  
        )
        if template_type == "md":
            self.model_card_md = self.build_model_card(template_type=template_type)
        else:
            self.model_card_html = self.build_model_card(template_type=template_type)
        return self

    def build_model_card(self, template_type):
        """Build graphics and add them to model card"""
        static = False
        if template_type == "md":
            static = True
        self.scaffold_assets()
        # Add Dataset Statistics
        if self.data_stats:
            add_dataset_feature_statistics_plots(self.model_card, self.data_stats.keys(), self.data_stats.values())
            for dataset in self.model_card.model_parameters.data:
                # Make sure graphs are ordered the same
                dataset.graphics.collection = sorted(dataset.graphics.collection, key=lambda x: x.name)
        # Add Evaluation Statistics
        if isinstance(self.metrics_by_threshold, pd.DataFrame):
            add_overview_graphs(self.model_card,
                                self.metrics_by_threshold,
                                static)
            add_eval_result_plots(self.model_card,
                                    self.metrics_by_threshold,
                                    static)
        if isinstance(self.metrics_by_group, pd.DataFrame):
            add_eval_result_slicing_metrics(self.model_card, self.metrics_by_group, static)
        self.update_model_card(self.model_card)
        return self.export_format(self.model_card)

    def update_model_card(self, model_card: ModelCard) -> None:
        """Updates the JSON file in the MCT assets directory.

        Args:
          model_card (ModelCard): The updated model card to write back.

        Raises:
          ValidationError: when the given model_card is invalid with reference to the schema.
        """
        self._write_json_file(self._mc_json_file, model_card)

    def scaffold_assets(self):
        """Generates the Model Card Tookit assets.

        Assets include the ModelCard JSON file, Model Card document, and jinja
        template. These are written to the `output_dir` declared at
        initialization.

        An assets directory is created if one does not already exist.

        If the MCT is initialized with a `mlmd_store`, it further auto-populates
        the model cards properties as well as generating related plots such as model
        performance and data distributions.

        Returns:
            A ModelCard representing the given model.

        Raises:
            FileNotFoundError: on failure to copy the template files.
        """

        # Write JSON file for model card
        self._write_json_file(self._mc_json_file, self.model_card)

        # Write UI template files.
        for template_path in _UI_TEMPLATES[self.template_type]:
            template_content = pkgutil.get_data("intel_ai_safety.model_card_gen", template_path)
            if template_content is None:
                raise FileNotFoundError(f"Cannot find file: '{template_path}'")
            template_content = template_content.decode("utf8")
            self._write_file(os.path.join(self.output_dir, template_path), template_content)

    def _jinja_loader(self, template_dir: Text) -> jinja2.FileSystemLoader:
        return jinja2.FileSystemLoader(template_dir)

    def _write_file(self, path: Text, content: Text) -> None:
        """Write content to the path."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w+") as f:
            f.write(content)

    def _write_json_file(self, path: Text, model_card: ModelCard) -> None:
        """Write serialized model card JSON to the path."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(model_card.to_json().encode())

    @staticmethod
    def _read_json_file(path: Text) -> Optional[ModelCard]:
        """Read serialized model card JSON from the path."""
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            model_card = ModelCardGen._read_json(f.read())
        return model_card

    @staticmethod
    def _read_json(model_card_json: Union[Dict[Text, Any], Text]) -> Optional[ModelCard]:
        """Read serialized model card proto from the path."""
        model_card = ModelCard()
        model_card.merge_from_json(model_card_json)
        return model_card

    def export_format(
        self,
        model_card: Optional[ModelCard] = None,
        template_path: Optional[Text] = None,
        output_file=None,
    ) -> Text:
        """Generates a model card document based on the MC assets.

        Args:
            model_card (ModelCard): The ModelCard object, generated from `scaffold_assets()`.
                If not provided, it will be read from the ModelCard proto file in the
                assets directory.
            template_path (str): The file path of the Jinja template. If not provided, the
                default template will be used.
            output_file (str): The file name of the generated model card. If not provided,
                the default 'model_card.html' will be used. If the file already exists,
                then it will be overwritten.

        Returns:
            The model card file content.
        """

        if template_path is None:
            template_path = os.path.join(self._mc_template_dir,_TEMPLATE_DIR)
        if output_file is None:
            output_file = f"model_card.{self.template_type}"
        template_dir = os.path.dirname(template_path)
        template_file = _DEFAULT_UI_TEMPLATE_FILE[self.template_type]

        # If model_card is passed in, write to JSON file.
        if model_card:
            self.update_model_card(model_card)
        # If model_card is not passed in, read from JSON file.
        else:
            model_card = ModelCardGen._read_json_file(self._mc_json_file)
        if not model_card:
            raise ValueError("model_card could not be found. " "Call scaffold_assets() to generate model_card.")

        # Generate Model Card.
        jinja_env = jinja2.Environment(
            loader=self._jinja_loader(template_dir), autoescape=True, auto_reload=True, cache_size=0
        )
        template = jinja_env.get_template(template_file)
        model_card_file_content = template.render(
            model_details=model_card.model_details,
            model_parameters=model_card.model_parameters,
            quantitative_analysis=model_card.quantitative_analysis,
            considerations=model_card.considerations,
        )

        # Write the model card document file and return its contents.
        mode_card_file_path = os.path.join(self._model_cards_dir, output_file)
        self._write_file(mode_card_file_path, model_card_file_content)
        return model_card_file_content

    def _repr_html_(self):
        return self.model_card_html
    
    def _repr_md_(self):
        return self.model_card_md

    def display_model_card(self):
        if self.template_type == "md":
            display(Markdown(self._repr_md_()))
        else:
            display(HTML(self._repr_html_()))

    def export_html(self, filename):
        with open(filename, "w") as f:
            f.write(self._repr_html_())

    def export_model_card(self, filename):
        with open(filename, "w") as f:
            if self.template_type == "md":
                f.write(self._repr_md_())
            else:
                f.write(self._repr_html_())