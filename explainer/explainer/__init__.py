"""module init for explainer cli"""
import os
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from importlib.util import spec_from_loader, module_from_spec, LazyLoader, find_spec
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Type, List

import yaml

from .api import Explainer
from .cli import ExplainerCLI, cli

EXT_YAML = '.yaml'


@dataclass
class TabularData:
    """TabularData structure"""
    name: str
    path: str
    data: Any


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
    """Extends the ModuleSpec to add ExplainerModuleSpec specific fields
    """

    def __init__(self, spec: ExplainerSpec, loader: Loader):
        super().__init__(spec.name, loader=loader)
        self.config = spec
        #for module in self.config.dependencies:
        #    self._lazy_import(module)

    def _lazy_import(self, name: str) -> ModuleSpec:
        """Calls importlib.util.LazyImport

        Args:
            name (str): name of the module to lazy import

        Returns:
            ModuleSpec: a dummy modulespec
        """
        spec = find_spec(name)
        loader = LazyLoader(spec.loader)
        spec.loader = loader
        module = module_from_spec(spec)
        sys.modules[name] = module
        loader.exec_module(module)
        return module

    def __repr__(self):
        args = [f"name={self.name}",
                f"loader={self.loader}"]
        if self.origin is not None:
            args.append(f"origin={self.origin}")
        if self.submodule_search_locations is not None:
            args.append(
                f"submodule_search_locations={self.submodule_search_locations}")
        if self.config is not None:
            args.append(f"config={self.config}")
        return f"{self.__class__.__name__}, ".join(args)


class ExplainerLoader(Loader):
    """Loads yaml files that hold ModuleSpec definitions
    """

    def __init__(self, full_path: str):
        self._full_path = full_path

    def create_module(self, spec: ModuleSpec) -> ModuleSpec:
        """Return a module to initialize and into which to load.

        Args:
            spec (ModuleSpec): Provided ModuleSpec

        Raises:
            ImportError: Unable to create a ModuleSpec

        Returns:
            ModuleSpec:
        """
        try:
            with open(self._full_path, encoding="UTF-8") as yaml_file:
                yamldata = yaml.load(yaml_file, Loader=self.get_yaml_loader())
                return ExplainerModuleSpec(yamldata, spec.loader)
        except Exception as error:
            raise ImportError from error

    def exec_module(self, _module: ModuleSpec):
        """needs to reify parts of _module.config and add them to _module

        Args:
            module (ModuleSpec): _description_

        Returns:
            _type_: _description_
        """
        return None

    def spec(self, loader: yaml.SafeLoader,
             node: yaml.nodes.MappingNode) -> ExplainerSpec:
        """Construct a ExplainerSpec"""
        return ExplainerSpec(**loader.construct_mapping(node))

    def get_yaml_loader(self) -> Type[yaml.SafeLoader]:
        """Returns the yaml loader for the tag !ExplainerSpec

        Returns:
            Type[SafeLoader]: !ExplainerSpec
        """
        loader = yaml.SafeLoader
        loader.add_constructor(
            "!ExplainerSpec", self.spec)
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
    ExplainerMetaPathFinder imports yaml files that define a
    explanation's model, data and python dependencies within a zip file
    """

    def find_spec(self, fullname: str, path: str, _target: ModuleType = None) -> ModuleSpec:
        """Returns ExplainerLoader if the path is a yaml file that includes a !ExplainerSpec tag

        Args:
            fullname (str): _description_
            path (str): _description_
            _target (ModuleType, optional): _description_. Defaults to None.

        Returns:
            ModuleSpec: _description_
        """
        mod_name = fullname.split('.')[-1]
        paths = path if path else [os.path.abspath(os.curdir)]
        for check_path in paths:
            full_path = os.path.join(check_path, mod_name + EXT_YAML)
            if os.path.exists(full_path):
                return spec_from_loader(fullname,  ExplainerLoader(full_path))
        return None


sys.meta_path.insert(0, ExplainerMetaPathFinder())
