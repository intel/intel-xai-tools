"""module init for explainer cli"""
import os
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from importlib.util import spec_from_loader
from types import ModuleType
from typing import Type, List

import yaml

from .api import Explainer
from .cli import ExplainerCLI, cli

EXT_YAML = '.yaml'


class ExplainerSpec:
    """An ExplainerSpec which holds name, data, model, dependencies
    """
    def __init__(self, name: str, data: str, dependencies: List[str], model: str):
        self.name = name
        self.data = data
        self.dependencies = dependencies
        self.model = model

    def __repr__(self):
        return f"name={self.name} data={self.data} "\
            f"dependencies={self.dependencies} model={self.model}"

class ExplainerModuleSpec(ModuleSpec):
    def __init__(self, spec: ExplainerSpec, loader):
        super().__init__(spec.name, loader=loader)
        self.spec = spec
        #self.data = spec.data
        #self.model = spec.m

class ExplainerLoader(Loader):
    """Loads yaml files that hold ModuleSpec definitions
    """

    def __init__(self, full_path: str):
        self._full_path = full_path
        self._data = None

    def create_module(self, spec: ModuleSpec) -> ModuleSpec | None:
        try:
            with open(self._full_path, encoding="UTF-8") as yaml_file:
                yamldata = yaml.load(yaml_file, Loader=self.get_loader())
                return ExplainerModuleSpec(yamldata, spec.loader)
        except Exception as error:
            raise ImportError from error

    def exec_module(self, module: ModuleSpec):
        """_summary_

        Args:
            module (ModuleSpec): _description_

        Returns:
            _type_: _description_
        """
        module.__dict__.update({"data": self._data})
        return None

    def explainerspec_constructor(self, loader: yaml.SafeLoader,
                                  node: yaml.nodes.MappingNode) -> ExplainerSpec:
        """Construct a ExplainerSpec"""
        return ExplainerSpec(**loader.construct_mapping(node))

    def get_loader(self) -> Type[yaml.SafeLoader]:
        """_summary

        Returns:
            Type[SafeLoader]: _description_
        """
        loader = yaml.SafeLoader
        loader.add_constructor(
            "!ExplainerSpec", self.explainerspec_constructor)
        return loader

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


# population.py

# import csv
# from importlib import resources

# import matplotlib.pyplot as plt

# class _Population:
#     def __init__(self):
#         """Prepare to read the population file"""
#         self._data = {}
#         self.variant = "Medium"

#     @property
#     def data(self):
#         """Read data from disk"""
#         if self._data:  # Data has already been read, return it directly
#             return self._data

#         # Read data and store it in self._data
#         print(f"Reading population data for {self.variant} scenario")
#         with resources.open_text(
#             "data", "WPP2019_TotalPopulationBySex.csv"
#         ) as fid:
#             rows = csv.DictReader(fid)

#             # Read data, filter the correct variant
#             for row in rows:
#                 if int(row["LocID"]) >= 900 or row["Variant"] != self.variant:
#                     continue

#                 country = self._data.setdefault(row["Location"], {})
#                 population = float(row["PopTotal"]) * 1000
#                 country[int(row["Time"])] = round(population)
#         return self._data

#     def get_country(self, country):
#         """Get population data for one country"""
#         country = self.data[country]
#         years, population = zip(*country.items())
#         return years, population

#     def plot_country(self, country):
#         """Plot data for one country, population in millions"""
#         years, population = self.get_country(country)
#         plt.plot(years, [p / 1e6 for p in population], label=country)

#     def order_countries(self, year):
#         """Sort countries by population in decreasing order"""
#         countries = {c: self.data[c][year] for c in self.data}
#         return sorted(countries, key=lambda c: countries[c], reverse=True)

# # Instantiate the singleton
# data = _Population()


class ExplainerMetaPathFinder(MetaPathFinder):
    """_summary_

    Args:
        MetaPathFinder (_type_): _description_
    """

    def find_spec(self, fullname: str, path: str, _target: ModuleType = None) -> ModuleSpec | None:
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
