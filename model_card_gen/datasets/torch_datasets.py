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

from numpy.typing import NDArray as Array
from typing import Any
from model_card_gen.datasets import BaseDataset
import random
TorchTensorType = Any
try:
    import torch
    from torch.utils.data import DataLoader as loader
    from torch.utils.data import random_split, TensorDataset
    from torch import TensorType as TorchTensorType
except: ImportError

class PytorchDataset(BaseDataset):
    """
    Wrapper class for torch.utils.data.Dataset objects
    """
    def __init__(self, dataset, name="", feature_names=[]):
        self._dataset = dataset
        self._name = name
        self._feature_names = feature_names
        self._framework = "pytorch"
    
    @property
    def dataset(self):
        """
        Returns the framework dataset object (torch.utils.data.Dataset)
        """
        return self._dataset
    
    @property
    def data_loader(self):
        """
        A data loader object corresponding to the dataset
        """
        return self._data_loader

    @property
    def feature_names(self):
        """
        Returns the feature_names for dataset
        """
        return self._feature_names
    
    @property
    def framework(self):
        """
        Returns the framework for dataset
        """
        return self._framework

    def _make_subsets(lengths, names=['train', 'test']):
        """
        Split PytorchDataset into multiple PytorchDataset objects
        """
        subsets = random_split(self._dataset, lengths)
        return [PytorchDataset(subset, name) for subset, name in zip(subsets, names)]
        
class PytorchNumpyDataset(PytorchDataset):
    """
    Wrapper for creating PytorchDataset form numpy arrays
    """

    def __init__(self,
                 input_array : Array,
                 target_array : Array,
                 input_tensor_type: TorchTensorType=None,
                 name : str='',
                feature_names : list = []):
        dataset = self._make_dataset(input_array, target_array, input_tensor_type=input_tensor_type)
        super().__init__(dataset, name, feature_names)
        self._data_loader = self._make_dataloader()
        
    @property
    def input_tensors(self):
        """
        Returns the input tensors form dataset
        """
        return loader([input_tensor for input_tensor, _ in self.dataset])
    
    @property
    def target_tensors(self):
        """
        Returns the target tensors form dataset
        """
        return loader([target_tensor for _, target_tensor in self.dataset])
        
    def _make_dataset(self,
                      input_array: Array,
                      target_array: Array,
                      input_tensor_type: TorchTensorType=None):

        input_tensor = torch.from_numpy(input_array)
        if input_tensor_type:
            input_tensor = input_tensor.type(input_tensor_type)
        label_tensor = torch.from_numpy(target_array)
        return TensorDataset(input_tensor, label_tensor)
    
    def _make_dataloader(self, batch_size=1, num_workers=1, shuffle=True, generator=None):
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        return loader(self.dataset,
                      batch_size=batch_size,
                      shuffle=shuffle, 
                      num_workers=num_workers,
                      worker_init_fn=seed_worker,
                      generator=generator)