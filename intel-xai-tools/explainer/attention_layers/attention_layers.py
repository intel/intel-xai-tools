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

class Layer:
  def __init__(self, model):
    import ecco
    self.lm = ecco.from_pretrained(model, activations=True)

  def __call__(self, **kwargs):
    pass

class Activations(Layer):
  def __init__(self, model: str, text: str):
    super().__init__(model)
    self.inputs = self.lm.tokenizer([text], return_tensors="pt")
    self.output = self.lm(self.inputs)

  def __call__(self, **kwargs):
    self.output.run_nmf(**kwargs).explore()


class AttentionHeadView:
  def __init__(self, model):
    from transformers import AutoTokenizer, AutoModel
    self.model = AutoModel.from_pretrained(model, output_attentions=True)
    self.tokenizer = AutoTokenizer.from_pretrained(model)
  def __call__(self, text):
    from bertviz import head_view
    inputs = self.tokenizer.encode(text, return_tensors='pt')
    outputs = self.model(inputs)
    attention = outputs[-1]
    tokens = self.tokenizer.convert_ids_to_tokens(inputs[0])
    head_view(attention, tokens)


class AttentionModelView:
  def __init__(self, model):
    from transformers import AutoTokenizer, AutoModel
    self.model = AutoModel.from_pretrained(model, output_attentions=True)
    self.tokenizer = AutoTokenizer.from_pretrained(model)

  def __call__(self, sentence_a, sentence_b):
    from bertviz import model_view
    encoder_input_ids = self.tokenizer(sentence_a, return_tensors="pt", add_special_tokens=True).input_ids
    decoder_input_ids = self.tokenizer(sentence_b, return_tensors="pt", add_special_tokens=True).input_ids
    outputs = self.model(input_ids=encoder_input_ids, decoder_input_ids=decoder_input_ids)
    encoder_text = self.tokenizer.convert_ids_to_tokens(encoder_input_ids[0])
    decoder_text = self.tokenizer.convert_ids_to_tokens(decoder_input_ids[0])
    model_view(
      encoder_attention=outputs.encoder_attentions,
      decoder_attention=outputs.decoder_attentions,
      cross_attention=outputs.cross_attentions,
      encoder_tokens= encoder_text,
      decoder_tokens=decoder_text
    )

def activations(model: str, text: str):
  """
     Provides visuals into what layers are activating tokens within the text segment

     Args:
       model (pytorch): large language model such as bert, gpt-2
       text  (str): input text that is given to the model tokenizer

     Returns:
       LayerActivations: this class shows activations within the model layers
  """
  return Activations(model, text)

def attention_head_view(model):
  """
     visualize attention importance within one or more attention head layers
 
     Usage:
     - Hover over any token on the left/right side of the visualization to filter attention from/to that token. The colors correspond to different attention heads.
     - Double-click on any of the colored tiles at the top to filter to the corresponding attention head.
     - Single-click on any of the colored tiles to toggle selection of the corresponding attention head.
     - Click on the Layer drop-down to change the model layer (zero-indexed).
     - The lines show the attention from each token (left) to every other token (right). Darker lines indicate higher attention weights. 
     - When multiple heads are selected, the attention weights are overlaid on one another.
 
     Args:
       model (pytorch): large language model such as bert, gpt-2
       text  (str): input text that is given to the model tokenizer
 
     Returns:
       Activations: this class shows activations within the model layers
  """
  return AttentionHeadView(model)

def attention_model_view(model):
  """
     shows token importance within the self-attention model view

     Args:
       model (pytorch): large language model such as bert, gpt-2
       text  (str): input text that is given to the model tokenizer

     Returns:
       Activations: this class shows activations within the model layers
  """
  return AttentionModelView(model)

def attention_neuron_view(model):
  """
     shows token importance within the self-attention neuron view

     Args:
       model (pytorch): large language model such as bert, gpt-2

     Returns:
       Activations: this class shows attention heads within the model layers

     Reference:
       https://github.com/jessevig/bertviz/blob/master/notebooks/neuron_view_bert.ipynb
  """
  class AttentionNeuronView:
    def __init__(self, model):
      from bertviz.transformers_neuron_view import BertModel, BertTokenizer
      do_lower_case = True
      if type(model) is str:
          self.model = BertModel.from_pretrained(model)
          self.tokenizer = BertTokenizer.from_pretrained(model, do_lower_case=do_lower_case)
      else:
          self.model = model
          model_name = model.name_or_path.rsplit('/')[-1]
          self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)

    def __call__(self, *args):
      from bertviz.neuron_view import show
      model_type = 'bert'
      show(self.model, model_type, self.tokenizer, *args, display_mode='dark', layer=2, head=0)

  return AttentionNeuronView(model)

def attributions(model: str):
  class Attributions(Layer):
    def __init__(self, model: str):
      super().__init__(model)
    def __call__(self, text, **kwargs):
      return self.lm.generate(text, **kwargs)
  return Attributions(model)

def predictions(model: str, text: str):
  class Predictions(Layer):
    def __init__(self, model: str, text: str):
      super().__init__(model)
      self.output = self.lm.generate(text, generate=1, do_sample=False)
    def __call__(self, **kwargs):
      self.output.layer_predictions(**kwargs)
  return Predictions(model, text)

def rankings(model: str):
  class Rankings(Layer):
    def __init__(self, model: str):
      super().__init__(model)
    def __call__(self, text, **kwargs):
      return self.lm.generate(text, **kwargs)
  return Rankings(model)
