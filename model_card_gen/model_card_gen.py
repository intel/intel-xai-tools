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
This module allows users to classes for performing end to end workflows
around model card generation.
"""

#Base
import os
import sys
import pkgutil
import tempfile
# External
import jinja2
import pandas as pd
import tensorflow_data_validation as tfdv
from IPython.display import display, HTML
# Internal
from model_card_gen.analyze import get_analysis
from model_card_gen.model_card import ModelCard
from model_card_gen.graphics.add_graphics import (
    add_dataset_feature_statistics_plots,
    add_overview_graphs,
    add_eval_result_plots,
    add_eval_result_slicing_metrics)
# Typing
import tensorflow_model_analysis as tfma
from typing import Optional, Sequence, Text, Union, Dict, Any
DataFormat = Union[pd.DataFrame, Text]

_UI_TEMPLATES = (
    'template/html/default_template.html.jinja',
    'template/html/style/default_style.html.jinja',
    'template/html/macros/default_macros.html.jinja',
    'template/html/js/plotly_js_header.html.jinja',
)
_DEFAULT_UI_TEMPLATE_FILE = os.path.join('html', 'default_template.html.jinja')
_MC_JSON_FILE = os.path.join('data', 'model_card.json')
_TEMPLATE_DIR = 'template'
_MODEL_CARDS_DIR = 'model_cards'
_DEFAULT_MODEL_CARD_FILE_NAME = 'model_card.html'

class ModelCardGen():
    """Generate ModelCard from with TFMA

    Args:
        data_sets (dict): dictionary with keys of name of dataset and value to path
        model_path (str): representing TF SavedModel path
        eval_config (tfma.EvalConfig or str) : tfma config object or string to config file path
        model_card (ModelCard or dict): pre-generated ModelCard Python object or dictionary following model card schema
        eval_results (tfma.EvalResults): pre-generated tfma results for when you do not wish to run evaluator
        output_dir (str): representing of where to output model card
    """

    def __init__(self,
                 data_sets: Dict[Text, Text] = {},
                 model_path: Text = '',
                 eval_config: Union[tfma.EvalConfig, str] = None,
                 model_card: Union[ModelCard, Dict[Text, Any]] = None,
                 eval_results: Sequence[tfma.EvalResult] = None,
                 output_dir: Text = ''):

        self.data_sets = self.check_data_sets(data_sets)
        self.model_path = model_path
        self.eval_config = eval_config
        # Local asset paths
        self.output_dir = output_dir or tempfile.mkdtemp()
        self._mc_json_file = os.path.join(self.output_dir, _MC_JSON_FILE)
        self._mc_template_dir = os.path.join(self.output_dir, _TEMPLATE_DIR)
        self._model_cards_dir = os.path.join(self.output_dir, _MODEL_CARDS_DIR)
        if isinstance(model_card, ModelCard):
          self.model_card = model_card
        elif isinstance(model_card, dict) or isinstance(model_card, str):
          self.model_card = ModelCardGen._read_json(model_card)
        else:
          self.model_card = ModelCard()
        # Generated Attributes
        self.data_stats = None
        self.eval_results = eval_results
        self.model_card_html = ''

    @classmethod
    def generate(cls,
                 data_sets: Dict[Text, DataFormat],
                 eval_config: Union[tfma.EvalConfig, str],
                 model_path: Text = '',
                 model_card: Union[ModelCard, Dict[Text, Any], Text] = None,
                 output_dir: Text = ''):
        """Class Factory starting TFMA analysis and generating ModelCard

        Args:
            data_sets (dict): dictionary with keys of name of dataset and value to path
            model_path (str): representing TF SavedModel path
            eval_config (tfma.EvalConfig or str) : tfma config object or string to config file path
            model_card (ModelCard or dict): pre-generated ModelCard Python object or dictionary following model card schema 
            output_dir (str): representing of where to output model card

        Returns:
            ModelCardGen

        Raises:
            ValueError: when invalid value for data_sets argument is empty
            TypeError: when data_sets argument is  not type dict

        Example:
            >>> from model_card_gen.model_card_gen import ModelCardGen
            >>> model_path = 'compas/model'
            >>> data_paths = {'eval': 'compas/eval.tfrecord', 'train': 'compas/train.tfrecord'}
            >>> eval_config = 'compas/eval_config.proto'
            >>> mcg = ModelCardGen.generate(data_paths, model_path, eval_config) #doctest:+SKIP
        """
        self = cls(
            data_sets,
            model_path,
            eval_config,
            model_card=model_card,
            output_dir=output_dir)
        self.data_stats = self.get_stats()
        self.eval_results = get_analysis(model_path=model_path,
                                         eval_config=eval_config,
                                         datasets=data_sets,)
        self.model_card_html = self.build_model_card()
        return self
    
    @classmethod
    def tf_generate(cls,
                 data_sets: Dict[Text, DataFormat],
                 eval_config: Union[tfma.EvalConfig, str],
                 model_path: Text = '',
                 model_card: Union[ModelCard, Dict[Text, Any], Text] = None,
                 output_dir: Text = ''):
        """Class Factory starting TFMA analysis and generating ModelCard

        Args:
            data_sets (dict): dictionary with keys of name of dataset and value to path
            model_path (str): representing TF SavedModel path
            eval_config (tfma.EvalConfig or str) : tfma config object or string to config file path
            model_card (ModelCard or dict): pre-generated ModelCard Python object or dictionary following model card schema 
            output_dir (str): representing of where to output model card

        Returns:
            ModelCardGen

        Raises:
            ValueError: when invalid value for data_sets argument is empty
            TypeError: when data_sets argument is  not type dict
        """
        from model_card_gen.analyze import TFAnalyzer

        self = cls(
            data_sets,
            model_path,
            eval_config,
            model_card=model_card,
            output_dir=output_dir)
        self.data_stats = self.get_stats()
        self.eval_results = [
            TFAnalyzer.analyze(
                model_path=model_path,
                dataset=dataset,
                eval_config=eval_config)
            for dataset in data_sets.values()]
        self.model_card_html = self.build_model_card()
        return self

    @classmethod
    def pt_generate(cls,
                 data_sets: Dict[Text, DataFormat],
                 eval_config: Union[tfma.EvalConfig, str],
                 model_path: Text = '',
                 model_card: Union[ModelCard, Dict[Text, Any], Text] = None,
                 output_dir: Text = ''):
        """Class Factory starting TFMA analysis and generating ModelCard

        Args:
            data_sets (dict): dictionary with keys of name of dataset and value to path
            model_path (str): representing TF SavedModel path
            eval_config (tfma.EvalConfig or str) : tfma config object or string to config file path
            model_card (ModelCard or dict): pre-generated ModelCard Python object or dictionary following model card schema 
            output_dir (str): representing of where to output model card

        Returns:
            ModelCardGen

        Raises:
            ValueError: when invalid value for data_sets argument is empty
            TypeError: when data_sets argument is  not type dict
        """
        from model_card_gen.analyze import PTAnalyzer

        self = cls(
            data_sets,
            model_path,
            eval_config,
            model_card=model_card,
            output_dir=output_dir)
        self.data_stats = self.get_stats()
        self.eval_results = [
            PTAnalyzer.analyze(
                model_path=model_path,
                dataset=dataset,
                eval_config=eval_config)
            for dataset in data_sets.values()]
        self.model_card_html = self.build_model_card()
        return self
    
    def check_data_sets(self, data_sets):
        """Checks whether data_set object is not empty or not of type dict"""
        if not data_sets:
            raise ValueError("ModelCardGen revieved invalid value for data_sets argument")
        if not isinstance(data_sets, dict):
            raise TypeError("ModelCardGen requires data_sets argument to be of type dict")
        return data_sets

    def get_stats(self):
        """Get statistics_pb2.DatasetFeatureStatisticsList object for each dataset.
        Store each result in dictionary with corresponding name of dataset as key.
        """
        if all(isinstance(elem, str) for elem in self.data_sets.values()):
            return {name: tfdv.generate_statistics_from_tfrecord(data_location=path)
                        for name, path in self.data_sets.items()}
        elif all(isinstance(elem, pd.DataFrame) for elem in self.data_sets.values()):
            return {name: tfdv.generate_statistics_from_dataframe(path)
                    for name, path in self.data_sets.items()}

    def build_model_card(self):
        """Build graphics and add them to model card"""
        self.scaffold_assets()
        # Add Dataset Statistics
        if self.data_stats:
            add_dataset_feature_statistics_plots(self.model_card, self.data_stats)
            for dataset in self.model_card.model_parameters.data:
                # Make sure graphs are ordered the same
                dataset.graphics.collection = sorted(dataset.graphics.collection,
                    key=lambda x: x.name)
        # Add Evaluation Statistics
        for eval_result, dataset_name in zip(self.eval_results, self.data_sets.keys()):
            add_overview_graphs(self.model_card, eval_result, dataset_name)
            add_eval_result_plots(self.model_card, eval_result)
            add_eval_result_slicing_metrics(self.model_card, eval_result)
        self.update_model_card(self.model_card)
        return self.export_format(self.model_card)


    def update_model_card(self, model_card: ModelCard) -> None:
      """Updates the JSON file in the MCT assets directory.
  
      Args:
        model_card (ModelCard): The updated model card to write back.
  
      Raises:
        Error: when the given model_card is invalid with reference to the schema.
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
        for template_path in _UI_TEMPLATES:
            template_content = pkgutil.get_data('model_card_gen', template_path)
            if template_content is None:
                raise FileNotFoundError(f"Cannot find file: '{template_path}'")
            template_content = template_content.decode('utf8')
            self._write_file(
                os.path.join(self.output_dir, template_path), template_content)

    def _jinja_loader(self, template_dir: Text) -> jinja2.FileSystemLoader:
        return jinja2.FileSystemLoader(template_dir)

    def _write_file(self, path: Text, content: Text) -> None:
        """Write content to the path."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w+') as f:
            f.write(content)
  
    def _write_json_file(self, path: Text, model_card: ModelCard) -> None:
        """Write serialized model card JSON to the path."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(model_card.to_json().encode())

    @staticmethod
    def _read_json_file(path: Text) -> Optional[ModelCard]:
        """Read serialized model card JSON from the path."""
        if not os.path.exists(path):
            return None
        with open(path, 'r') as f:
            model_card = ModelCardGen._read_json(f.read())
        return model_card

    @staticmethod
    def _read_json(model_card_json: Union[Dict[Text, Any], Text]) -> Optional[ModelCard]:
        """Read serialized model card proto from the path."""
        model_card = ModelCard()
        model_card.merge_from_json(model_card_json)
        return model_card

    def export_format(self, model_card: Optional[ModelCard] = None,
                      template_path: Optional[Text] = None,
                      output_file=_DEFAULT_MODEL_CARD_FILE_NAME) -> Text:
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

        if not template_path:
            template_path = os.path.join(self._mc_template_dir,
                                         _DEFAULT_UI_TEMPLATE_FILE)
        template_dir = os.path.dirname(template_path)
        template_file = os.path.basename(template_path)
  
      # If model_card is passed in, write to JSON file.
        if model_card:
            self.update_model_card(model_card)
      # If model_card is not passed in, read from JSON file.
        else:
            model_card = ModelCardGen._read_json_file(self._mc_json_file)
        if not model_card:
            raise ValueError(
                'model_card could not be found. '
                'Call scaffold_assets() to generate model_card.')
  
        # Generate Model Card.
        jinja_env = jinja2.Environment(
          loader=self._jinja_loader(template_dir),
          autoescape=True,
          auto_reload=True,
          cache_size=0)
        template = jinja_env.get_template(template_file)
        model_card_file_content = template.render(
            model_details=model_card.model_details,
            model_parameters=model_card.model_parameters,
            quantitative_analysis=model_card.quantitative_analysis,
            considerations=model_card.considerations)
  
        # Write the model card document file and return its contents.
        mode_card_file_path = os.path.join(self._model_cards_dir, output_file)
        self._write_file(mode_card_file_path, model_card_file_content)
        return model_card_file_content
    
    def _repr_html_(self):
        return self.model_card_html

    def display_model_card(self):
        display(HTML(self._repr_html_()))
    
    def export_html(self, filename):
        with open(filename, 'w') as f:
            f.write(self._repr_html_())
