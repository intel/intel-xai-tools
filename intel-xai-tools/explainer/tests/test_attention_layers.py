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
### libraries to support tests ###
import pytest
import torch 
torch.manual_seed(0)
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['KMP_WARNINGS'] = 'off'
### library to be tested ###
from explainer import attention_layers 
###################################

device = torch.device('cpu')

def test_activations():
    model = 'bert-base-uncased'
    text = ''' 
    Now I ask you: what can be expected of man since he is a being endowed with strange qualities? 
    Shower upon him every earthly blessing, drown him in a sea of happiness, so that nothing but bubbles of bliss 
    can be seen on the surface; give him economic prosperity, such that he should have nothing else to do but sleep, 
    eat cakes and busy himself with the continuation of his species, and even then out of sheer ingratitude, sheer spite, 
    man would play you some nasty trick. He would even risk his cakes and would deliberately desire the most fatal rubbish, 
    the most uneconomical absurdity, simply to introduce into all this positive good sense his fatal fantastic element. 
    It is just his fantastic dreams, his vulgar folly that he will desire to retain, simply in order to prove to himself--as though that were so necessary-- 
    that men still are men and not the keys of a piano, which the laws of nature threaten to control so completely that soon one will be able to desire nothing but by the calendar. 
    And that is not all: even if man really were nothing but a piano-key, even if this were proved to him by natural science and mathematics, even then he would not become reasonable,
    but would purposely do something perverse out of simple ingratitude, simply to gain his point. And if he does not find means he will contrive destruction and chaos, will 
    contrive sufferings of all sorts, only to gain his point! He will launch a curse upon the world, and as only man can curse (it is his privilege, the primary distinction 
    between him and other animals), may be by his curse alone he will attain his object--that is, convince himself that he is a man and not a piano-key!
    '''

    activations = attention_layers.activations(model, text)
    assert isinstance(activations, attention_layers.Activations)
    
    try:
        activations(n_components=8)
    except ValueError:
        pytest.UsageError('Activations calculated but failed on visualizing')