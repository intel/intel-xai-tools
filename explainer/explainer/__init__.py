"""module init for explainer

This module init registers a finder and loader to
import yaml files and create a ModuleSpec
"""
import os
import sys
import zipfile
from dataclasses import dataclass
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from importlib.util import (LazyLoader, find_spec, module_from_spec,
                            spec_from_loader)
from types import ModuleType
from typing import Any, Callable, Type, TypedDict

import yaml
from yaml import YAMLError

EXT_YAML = '.yaml'


@dataclass
class TabularData:
    """TabularData structure"""
    name: str
    path: str
    data: Any


class ExplainerSpec(yaml.YAMLObject):
    """An ExplainerSpec which holds name, data, model, entry_point and plugin path
    """
    yaml_tag = "!ExplainerSpec"

    def __init__(self, name: str, plugin: str, dataset: str = None,
                 dependencies: list[str] = None, entry_point: str = None, model: str = None):
        self.name: str = name
        self.dataset: str = dataset
        self.dependencies: list[str] = dependencies
        self.entry_point: str = entry_point
        self.model: str = model
        self.plugin: str = plugin

    def __repr__(self):
        info = f'{self.__class__.__name__}(name="{self.name}"'
        if hasattr(self, 'dataset'):
            info += f', dataset="{self.dataset}"'
        if hasattr(self, 'dependencies'):
            info += f', dependencies="{self.dependencies}"'
        if hasattr(self, 'entry_point'):
            info += f', entry_point="{self.entry_point}"'
        if hasattr(self, 'model'):
            info += f', model="{self.model}"'
        if hasattr(self, 'plugin'):
            info += f', plugin="{self.plugin}"'
        info += ")"
        return info


class ExplainerModuleSpec(ModuleSpec):
    """Extends the ModuleSpec to add ExplainerModuleSpec specific fields
    """

    def __init__(self, spec: ExplainerSpec, loader: Loader):
        super().__init__(spec.name, loader=loader)
        self.spec = spec
        # for module in self.spec.dependencies:
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
        info = f'{self.name}(name="{self.name}"'
        if hasattr(self, 'loader'):
            info += f', loader="{self.loader}"'
        if hasattr(self, 'origin'):
            info += f', origin="{self.origin}"'
        if hasattr(self, 'submodule_search_locations'):
            info += f', submodule_search_locations="{self.submodule_search_locations}"'
        if hasattr(self, 'spec'):
            info += f', spec="{self.spec}"'
        return info


class ExplainerLoader(Loader):
    """Loads yaml files that hold ExplainerSpec definitions
    """

    def __init__(self, full_path: str):
        self._full_path = full_path

    def create_module(self, spec: ModuleSpec=None) -> ModuleSpec:
        """Return a module to initialize and into which to load.

        Args:
            spec (ModuleSpec): Provided ModuleSpec

        Raises:
            ImportError: Unable to create a ModuleSpec

        Returns:
            ModuleSpec:
        """
        loader = self
        module: ModuleSpec = None
        if spec is not None and hasattr(spec, 'loader'):
            loader = spec.loader
        try:
            with open(self._full_path, encoding="UTF-8") as yaml_file:
                yamlspec = yaml.load(yaml_file, self.get_yaml_loader())
                module = ExplainerModuleSpec(yamlspec, loader)
                spec=module.spec
                if hasattr(spec, "plugin"):
                    zipname=spec.plugin
                    zippath=os.path.join(os.path.dirname(self._full_path), zipname)
                    dirname=os.path.splitext(zippath)[0]
                    print(f"ExplainerLoader.create_module zippath={zippath} dirname={dirname}")
                    if not os.path.exists(dirname) and os.path.exists(zippath):
                        with zipfile.ZipFile(zippath, mode="r") as archive:
                            print(f"extracting all to {dirname}")
                            archive.extractall(dirname)
                            os.remove(zippath)
                    if os.path.exists(dirname) and dirname not in sys.path:
                        print(f"*** inserting into sys.path {dirname} ***")
                        sys.path.insert(0, dirname)
        except YAMLError as exc:
            if hasattr(exc, 'problem_mark'):
                mark = exc.problem_mark
                print(f"Error position: ({mark.line+1}:{mark.column+1})")
        except Exception as error:
            raise ImportError from error
        return module

    def exec_module(self, _module: ModuleSpec):
        """needs to reify parts of _module.spec and add them to _module

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
        loader = yaml.FullLoader
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
    """ExplainerMetaPathFinder imports yaml files that define a
       explanation's model, data, entry_point and plugin path
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


class MonkeypatchOnImportHook(MetaPathFinder, Loader):
    """imports a module dyanmically
    """
    def __init__(self, module_to_monkeypatch):
        self._modules_to_monkeypatch: TypedDict[str, Callable[[Any], None]] = module_to_monkeypatch
        self._in_create_module = False

    def find_module(self, fullname, path=None):
        spec = self.find_spec(fullname, path)
        if spec is None:
            return None
        return spec

    def create_module(self, spec: ModuleSpec):
        self._in_create_module = True

        real_spec = find_spec(spec.name)

        real_module = module_from_spec(real_spec)
        real_spec.loader.exec_module(real_module)

        self._modules_to_monkeypatch[spec.name](real_module)

        self._in_create_module = False
        return real_module

    def exec_module(self, module: ModuleType):
        """inherits from Loader

        Args:
            module (_type_): _description_
        """
        try:
            _ = sys.modules.pop(module.__name__)
        except KeyError:
            print(f"module {module.__name__} is not in sys.modules")
        sys.modules[module.__name__] = module
        globals()[module.__name__] = module

    def find_spec(self, fullname: str, _path=None, _target=None):
        """finds the ModuleSpec

        Args:
            fullname (str): fullname of module
            _path (_type_, optional): _description_. Defaults to None.
            _target (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if fullname not in self._modules_to_monkeypatch:
            return None

        if self._in_create_module:
            # if we are in the create_module function,
            # we return the real module (return None)
            return None

        spec = ModuleSpec(fullname, self)
        return spec

def test_perform_mpl_monkeypatch():
    """test function
    """
    def perform_mpl_monkeypatch(_pyplot):
        print(f"Monkeypatching pyplot {_pyplot}")
    # Install monkeypatcher
    sys.meta_path.insert(
        0, MonkeypatchOnImportHook({
            "matplotlib.pyplot": perform_mpl_monkeypatch
        })
    )
    import matplotlib.pyplot


sys.meta_path.insert(0, ExplainerMetaPathFinder())
