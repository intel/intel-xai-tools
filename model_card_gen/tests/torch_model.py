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

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset
import tempfile

COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race", "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "country",
    "label"
] 

CATEGORICAL_FEATURE_KEYS = [
    'workclass',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
]

DROP_COLUMNS = ['fnlwgt', 'education']
LABEL_KEY = 'label'

class AdultNN(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.linear1 = nn.Linear(feature_size, feature_size)
        self.sigmoid1 = nn.Sigmoid()
        self.linear2 = nn.Linear(feature_size, 8)
        self.sigmoid2 = nn.Sigmoid()
        self.linear3 = nn.Linear(8, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        lin1_out = self.linear1(x)
        sigmoid_out1 = self.sigmoid1(lin1_out)
        sigmoid_out2 = self.sigmoid2(self.linear2(sigmoid_out1))
        return self.softmax(self.linear3(sigmoid_out2))

def get_data():
    from sklearn.datasets import fetch_openml
    data = fetch_openml(data_id=1590, as_frame=True)
    raw_data = data.data
    raw_data[LABEL_KEY] = data.target
    adult_data = raw_data.copy()
    adult_data = adult_data.drop(DROP_COLUMNS, axis=1)
    adult_data = pd.get_dummies(adult_data, columns=CATEGORICAL_FEATURE_KEYS)
    adult_data[LABEL_KEY] = adult_data[LABEL_KEY].map({'<=50K': 0, '>50K': 1})
    feature_names = list(adult_data.drop([LABEL_KEY], axis=1).columns)
    y = adult_data[LABEL_KEY].to_numpy()
    X = adult_data.drop([LABEL_KEY], axis=1).to_numpy()

    return TensorDataset(torch.Tensor(X).type(torch.FloatTensor),
                         torch.Tensor(y).type(torch.LongTensor)) , feature_names

def get_trained_model(adult_dataset, feature_names):
    _, tmp_train = tempfile.mkstemp()
    net = AdultNN(len(feature_names))
    criterion = nn.CrossEntropyLoss()
    num_epochs = 200

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    input_tensor, label_tensor = adult_dataset[:]
    for epoch in range(num_epochs):
        output = net(input_tensor)
        loss = criterion(output, label_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print ('Epoch {}/{} => Loss: {:.2f}'.format(epoch+1, num_epochs, loss.item()))
    torch.jit.save(torch.jit.script(net), tmp_train)
    return tmp_train
