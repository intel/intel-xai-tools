"""module init for explainer cli"""
import json
import os
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from importlib.util import spec_from_loader
from types import ModuleType

from .api import Explainer
from .cli import ExplainerCLI, cli

EXT_YAML = '.yaml'

__all__ = []


class ExplainerLoader(Loader):
    """Loads yaml files that hold ModuleSpec definitions
    """

    def __init__(self, full_path: str):
        self._full_path = full_path
        self._data = None

    def create_module(self, spec):
        try:
            with open(self._full_path, encoding="UTF-8") as json_file:
                self._data = json.load(json_file)
        except Exception as error:
            raise ImportError from error
        return None

    def exec_module(self, module: ModuleSpec):
        """_summary_

        Args:
            module (ModuleSpec): _description_

        Returns:
            _type_: _description_
        """
        module.__dict__.update({"data": self._data})
        return None


class ExplainerMetaPathFinder(MetaPathFinder):
    """_summary_

    Args:
        MetaPathFinder (_type_): _description_
    """
    def find_spec(self, fullname: str, path: str, _target: ModuleType = None) -> ModuleSpec|None:
        """_summary_

        Args:
            fullname (str): _description_
            path (str): _description_
            _target (ModuleType, optional): _description_. Defaults to None.

        Returns:
            ModuleSpec|None: _description_
        """
        mod_name = fullname.split('.')[-1]
        paths = path if path else [os.path.abspath(os.curdir)]
        for check_path in paths:
            full_path = os.path.join(check_path, mod_name + EXT_YAML)
            if os.path.exists(full_path):
                return spec_from_loader(fullname,  ExplainerLoader(full_path))
        return None


sys.meta_path.insert(0, ExplainerMetaPathFinder())
