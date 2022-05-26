"""module init for explainer cli"""
import os
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from importlib.util import spec_from_loader
from types import ModuleType

import yaml

from .api import Explainer
from .cli import ExplainerCLI, cli

EXT_YAML = '.yaml'
class ExplainerLoader(Loader):
    """Loads yaml files that hold ModuleSpec definitions
    """

    def __init__(self, full_path: str):
        self._full_path = full_path
        self._data = None

    def create_module(self, spec):
        try:
            with open(self._full_path, encoding="UTF-8") as yaml_file:
                self._data = yaml.load(yaml_file, Loader=yaml.SafeLoader)
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

# import csv
# from importlib import resources

# def read_population_file(year, variant="Medium"):
#     population = {}

#     print(f"Reading population data for {year}, {variant} scenario")
#     with resources.open_text(
#         "data", "WPP2019_TotalPopulationBySex.csv"
#     ) as fid:
#         rows = csv.DictReader(fid)

#         # Read data, filter the correct year
#         for row in rows:
#             if row["Time"] == year and row["Variant"] == variant:
#                 pop = round(float(row["PopTotal"]) * 1000)
#                 population[row["Location"]] = pop

#     return population


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
