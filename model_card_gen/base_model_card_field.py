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

"""BaseModelCardField.

This class serves as a basic shared API between all Model Card data classes (
see model_card.py).
"""
import abc
import json as json_lib
from typing import Any, Dict, Text
import dataclasses
from model_card_gen import validation

class BaseModelCardField(abc.ABC):
    """Model card field base class.

    This is an abstract class. All the model card fields should inherit this class.
    """
    def _from_json(self, json_dict: Dict[str, Any],
                                 field: "BaseModelCardField") -> "BaseModelCardField":
        """Parses a JSON dictionary into the current object."""
        for subfield_key, subfield_json_value in json_dict.items():
            if subfield_key.startswith(validation.SCHEMA_VERSION_STRING):
                continue
            elif not hasattr(field, subfield_key):
                raise ValueError(
                        "BaseModelCardField %s has no such field named '%s.'" %
                        (field, subfield_key))
            elif isinstance(subfield_json_value, dict):
                subfield_value = self._from_json(
                        subfield_json_value, getattr(field, subfield_key))
            elif isinstance(subfield_json_value, list):
                subfield_value = []
                for item in subfield_json_value:
                    if isinstance(item, dict):
                        new_object = field.__annotations__[subfield_key].__args__[0]()  # pytype: disable=attribute-error
                        subfield_value.append(self._from_json(item, new_object))
                    else:  # if primitive
                        subfield_value.append(item)
            else:
                subfield_value = subfield_json_value
            setattr(field, subfield_key, subfield_value)
        return field

    def to_json(self) -> Text:
        """Convert this class object to json."""
        return json_lib.dumps(self.to_dict(), indent=2)

    def to_dict(self) -> Dict[Text, Any]:
        """Convert your model card to a python dictionary."""
        # ignore None properties recursively to allow missing values.
        ignore_none = lambda properties: {k: v for k, v in properties if v}
        return dataclasses.asdict(self, dict_factory=ignore_none)

    def clear(self):
        """Clear the subfields of this BaseModelCardField."""
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, BaseModelCardField):
                field_value.clear()
            elif isinstance(field_value, list):
                setattr(self, field_name, [])
            else:
                setattr(self, field_name, None)

    @classmethod
    def _get_type(cls, obj: Any):
        return type(obj)
