"""module init for explainer

This module init registers a finder and loader to
import yaml files and create a ModuleSpec
"""
import os
import csv
import sys
import re
import datetime

from dataclasses import dataclass
from importlib.metadata import entry_points as entrypoints
from importlib.abc import Loader, MetaPathFinder, ResourceReader, TraversableResources, Traversable
from importlib.machinery import ModuleSpec
from importlib.resources import open_text
from importlib.util import spec_from_loader
from types import ModuleType
from typing import Any, BinaryIO, Callable, Iterable, List, Sequence, Text, Type, Union

import yaml
from yaml import MarkedYAMLError

from .version import __version__

EXT_YAML = '.yaml'


@dataclass
class TabularData:
    """TabularData structure"""
    name: str
    path: str
    data: Any


class DataSetResource(TraversableResources):
    """Reads a dataset, inherits from TraversableResources
    """

    def __init__(self, path: str) -> None:
        print("DatSetResource.__init__")
        self._name: str = path
        self._data: DataSetResource

    @property
    def data(self):
        """Read data from from a path"""
        if self._data:  # Data has already been read, return it directly
            return self._data

        # Read data and store it in self._data
        with open_text("data", self._name) as fid:
            rows = csv.DictReader(fid)
            for row in rows:
                country = self._data.setdefault(row["Location"], {})
                population = float(row["PopTotal"]) * 1000
                country[int(row["Time"])] = round(population)
        return self._data

    def files(self) -> Traversable:
        """Return a Traversable object for the loaded package."""
        return None

    def open_resource(self, resource: Text) -> BinaryIO:
        return self.files().joinpath(resource).open('rb')

    def resource_path(self, resource: Text) -> Text:
        raise FileNotFoundError(resource)

    def is_resource(self, path: Text) -> bool:
        return self.files().joinpath(path).is_file()

    def contents(self) -> Iterable[str]:
        return (item.name for item in self.files().iterdir())


class ExplainerSpec(yaml.YAMLObject):
    """An ExplainerSpec which holds the following attributes:
    
    name, version, dataset, dependencies, entry_points, model and plugin
    """
    yaml_tag = "!ExplainerSpec"

    def __init__(self, name: str, version: Union[str,None]=None, dataset: Union[str,None]=None,
                 dependencies: Union[List[str],None]=None, entry_points: Union[List[str],None]=None,
                 model: Union[str,None]=None, plugin: Union[str,None]=None):
        self.name: str = name
        self.version: Union[str, None] = version
        self.dataset: Union[str, None] = dataset
        self.dependencies: Union[List[str],None] = dependencies
        self.entry_points: Union[List[str],None] = entry_points
        self.model: Union[str,None] = model
        self.plugin: Union[str,None] = plugin

    def __repr__(self):
        info = f'{self.__class__.__name__}:\n'
        info += f'\tname: {self.name}\n'
        if hasattr(self, 'version') and self.version is not None:
            info += f'\tversion: {self.version}\n'
        if hasattr(self, 'dataset') and self.dataset is not None:
            info += f'\tdataset: {self.dataset}\n'
        if hasattr(self, 'model') and self.model is not None:
            info += f'\tmodel: {self.model}\n'
        if hasattr(self, 'plugin') and self.plugin is not None:
            info += f'\tplugin: {self.plugin}\n'
        if hasattr(self, 'dependencies'):
            info += f'\tdependencies:\n'
            for dependency in self.dependencies:
                info += f'\t\t- {dependency}\n'
        if hasattr(self, 'entry_points'):
            info += f'\tentry_points:\n'
            for ep in self.entry_points:
                info += f'\t\t- "{ep}"\n'
        return info

@dataclass
class ExplainerYamlFile:
    tag: str
    name: str
    version: str
    plugin: str
    dependencies: list
    entry_points: list

    def __repr__(self):
        o = f"{self.tag}\nname: {self.name}\n"\
            f"version: {self.version}\n"\
            f"plugin: {self.plugin}\n"\
            f"dependencies: {self.dependencies}\n"\
            f"entry_points: {self.entry_points}\n"
        return o


class ExplainerYaml:
    """ExplainerYaml provides Explainer.create the yaml attributes needed to populate the yaml file
       where each attribute has an associated value. Each attribute's value can be retreived
       by iterating through the list of attributes and getting the associated value.
    """
  
    def __init__(self, name: str):
        plugin_name = re.sub('-', '_', name)
        version = "0.1.0"
        self.yaml_file = ExplainerYamlFile("--- !ExplainerSpec",
                                           name, 
                                           version,
                                           "explainer_explainers_"+plugin_name+"-"+version+"-py2.py3-none-any.whl",
                                           [],
                                           [])

    def create(self) -> str:
        return f"{self.yaml_file!r}"

@dataclass
class ExplainerPluginFile:
    name: str
    create: Callable
    overwrite: bool


class ExplainerPlugin:
    """ExplainerPlugin provides Explainer.generate the set of files for an Explainer plugin,
       where each filename has associated content. Each filename's content can be retreived 
       by iterating through the list of filenames and calling the associated content generator.
    """
    
    def __init__(self, spec: ExplainerSpec):
        self.spec = spec
        entry_point = re.sub('-', '_', self.spec.name)
        self.files: List[ExplainerPluginFile] = [
            ExplainerPluginFile("README.md", self.readme, False),
            ExplainerPluginFile(f"{entry_point}.py", self.entry_point, False),
            ExplainerPluginFile("Makefile", self.makefile, True),
            ExplainerPluginFile("setup.py", self.setup_py, True),
            ExplainerPluginFile("setup.cfg", self.setup_cfg, True),
            #ExplainerPluginFile"test/test.py": self.test, True)
        ]

    @property
    def autogenerated(self) -> str:
        now = datetime.datetime.now()
        return "AUTOGENERATED "+__version__

    def entry_point(self) -> str:
        header = "## API"
        entry_points = [header]
        eps = self.spec.entry_points
        if eps is not None:
            for entry_point in eps:
                parts: List[str] = entry_point.split(':')
                ep = parts[1]
                parts = ep.split()
                epname = parts[0]
                epdef = f"""def {epname}("""
                if len(parts) > 1:
                    epargs = parts[1]
                    result = re.search(r"[A-Za-z0-9_,\b\d]+", epargs)
                    if result is not None:
                        args = result.group().split(',')
                        if args is not None and len(args) > 0:
                            nargs = len(args) - 1
                            for arg in range(nargs):
                                epdef = epdef + args[arg] + ","
                            epdef = epdef + args[-1]
                epdef = epdef + "):\n    pass"
                entry_points.append(epdef)
        content = "\n\n".join(entry_points)
        return content

    def makefile(self) -> str:
        package_name = re.sub('-', '_', self.spec.name)
        return f"""# {self.autogenerated}\n
clean::
\t@rm -rf build dist explainer_explainers_{package_name}.egg-info test/plugins/{package_name}

wheel: clean
\tpython setup.py bdist_wheel

install: wheel
\tcd test && pip install ../dist/explainer_explainers_{package_name}-0.1-py2.py3-none-any.whl --target plugins/{package_name}

test: install
\tcd test && python test.py
"""

    def readme(self) -> str:
        header = "# "+self.spec.name
        subheaders = [
            "## Algorithms",
            "## Environment",
            "## Example",
            "## Toolkits",
            "## References"
        ]
        template = f"{header}\n\n"
        for i in subheaders:
            template += "\n".join([i]) + "\n\n"
        template += '\n'
        entry_points = [template]
        content = "\n\n".join(entry_points)
        return content

    def setup_py(self) -> str:
        sections = []
        package_name = re.sub('-', '_', self.spec.name)
        section = f"""from setuptools import setup

setup(
    name='explainer-explainers-{self.spec.name}',
    version='{self.spec.version}',
    zip_safe=False,
    platforms='any',
    py_modules=['{package_name}'],
    include_package_data=True,
    install_requires=[\n"""
        sections.append(section)
        dps = self.spec.dependencies
        if dps is not None:
            for dp in dps:
                sections.append("       '"+dp+"',\n")
        section = """   ],\n"""
        sections.append(section)
        section = f"""   entry_points="""
        sections.append(section+"{")
        section = f"""
        'explainer.explainers.{package_name}': [\n"""
        sections.append(section)
        eps = self.spec.entry_points
        if eps is not None:
            for ep in eps:
                sections.append("           '"+ep+"',\n")
        section_four = """       ]\n"""
        sections.append(section_four)
        section_five = """   },"""
        sections.append(section_five)
        section_six = f"""
    python_requires='>=3.9'
)
"""
        sections.append(section_six)
        content = " ".join(sections)
        return content
    
    def setup_cfg(self) -> str:
        return """[bdist_wheel]\nuniversal = 1\n"""

    def test(self) -> str:
        pass

class ExplainerModuleSpec(ModuleSpec):
    """Extends the ModuleSpec to add
       spec: ExplainerSpec
       entry_points: EntryPoints
       plugin_path: str
    """

    def __init__(self, spec: ExplainerSpec, loader: Union[Loader,None]=None, origin=None, loader_state=None,
                 is_package=None, verbose: bool=False):
        super().__init__(spec.name, loader=loader, origin=origin, loader_state=loader_state, is_package=is_package)
        self.spec: ExplainerSpec = spec
        self._verbose = verbose
        self.plugin_path = None
        if self._verbose is True:
            print(f"{self.__class__.__name__} __init__ {self!r}", file=sys.stderr)

    def __repr__(self):
        info = f'{self.__class__.__name__}(name="{self.name}"'
        if hasattr(self, 'loader'):
            info += f', loader="{self.loader}"'
        if hasattr(self, 'origin'):
            info += f', origin="{self.origin}"'
        if hasattr(self, 'submodule_search_locations'):
            info += f', submodule_search_locations="{self.submodule_search_locations}"'
        if hasattr(self, 'spec'):
            info += f', spec="{self.spec}"'
        if hasattr(self, 'plugin_path'):
            info += f', plugin_path="{self.plugin_path}"'
        if hasattr(self, '__package__'):
            info += f', __package__="{self.spec}"'
        info += ")"
        return info

    def __getitem__(self, key: str):
        name: str = re.sub('-', '_', self.name)
        group: str = f"explainer.explainers.{name}"
        if self._verbose is True:
            print(f"{self.__class__.__name__} __getitem__ group={group} name={name} key={key}", file=sys.stderr)        
        try:
            eps_list = entrypoints()
            eps = eps_list[group]
            for ep in eps:
                if ep.group == group and ep.name == key:
                    ep_func = ep.load()
                    if self._verbose is True:
                        print(f"{self.__class__.__name__} __getitem__ ep_func={ep_func}", file=sys.stderr) 
                    return ep_func
        except KeyError as ke:
            print(f"{self.__class__.__name__} KeyError group={group} name={name} key={key} {ke!r}", file=sys.stderr)
        except Exception as e:
            print(f"{self.__class__.__name__} Error group={group} name={name} key={key} {e!r}", file=sys.stderr)
        return None

    @property
    def entry_points(self) -> bool:
        name: str = re.sub('-', '_', self.name)
        group: str = f"explainer.explainers.{name}"
        try:
            eps_list = entrypoints()
            eps = eps_list[group]
            entry_points = [ep for ep in eps]
            for entry_point in entry_points:
                name = entry_point.name
                func = entry_point.load()
                setattr(self, name, func)
        except KeyError as k:
            print(f"{self.__class__.__name__} KeyError group={group} name={name} {k}", file=sys.stderr)
            return False
        except Exception as e:
            print(f"{self.__class__.__name__} Error group={group} name={name} {e}", file=sys.stderr)        
            return False
        return True

class ExplainerLoader(Loader):
    """Loads yaml files that hold ExplainerSpec definitions
    """
    def __init__(self, full_path: str, verbose: bool = False):
        if verbose is True:
            print(f"{self.__class__.__name__} __init__ full_path={full_path}", file=sys.stderr)
        if os.path.exists(full_path) is not True:
            print(f"{full_path}: does not exit", file=sys.stderr)
        self._full_path = full_path
        self._module: Union[ExplainerModuleSpec,None] = None
        self._loader: Union[Loader,None] = self
        self._verbose = verbose

    @property
    def module(self) -> Union[ExplainerModuleSpec,None]:
        """Creates an ExplainerModuleSpec

        Raises:
            ImportError: Error in yaml file

        Returns:
            ExplainerModuleSpec: subclass of ModuleSpec
        """
        if self._verbose is True:
            print(f"{self.__class__.__name__} module", file=sys.stderr)
        if self._module is None:
            self.create_module()
        return self._module

    def create_module(self, spec: Union[ModuleSpec,None]=None) -> Union[ModuleSpec,None]:
        """Return a module to initialize and into which to load.

        Args:
            spec (ModuleSpec): Provided ModuleSpec

        Raises:
            ImportError: Unable to create a ModuleSpec

        Returns:
            ExplainerModuleSpec:
        """
        if self._verbose is True:
            print(f"{self.__class__.__name__} create_module", file=sys.stderr)
        if spec is not None and hasattr(spec, 'loader'):
            self._loader = spec.loader
        if self._module is None:
            try:
                with open(self._full_path, encoding="UTF-8") as yaml_file:
                    yamlspec: ExplainerSpec = yaml.load(yaml_file, self.get_yaml_loader())
                    origin = None
                    loader_state = None
                    if spec is not None:
                        origin = spec.origin
                        loader_state = spec.loader_state
                    self._module = ExplainerModuleSpec(yamlspec, loader=self._loader, origin=origin,
                                                       loader_state=loader_state, is_package=True, verbose=self._verbose)
            except MarkedYAMLError as exc:
                if hasattr(exc, 'problem_mark'):
                    mark = exc.problem_mark
                    if mark is not None:
                        print(f"Error position: ({mark.line+1}:{mark.column+1})", file=sys.stderr)
            except Exception as error:
                if self._verbose == True:
                    print(f"throwing ImportError", file=sys.stderr)
                raise ImportError from error
            if self._module is not None:
                if hasattr(self._module.spec, "name"):
                    if self.load_plugin() is True:
                        self._module.entry_points
                if hasattr(self._module.spec, "dataset"):
                    self.load_dataset()
                if spec is not None:
                    if hasattr(spec, "origin"):
                        self._module.origin = spec.origin
                    if hasattr(spec, "submodule_search_locations"):
                        self._module.submodule_search_locations = spec.submodule_search_locations

        return self._module

    def exec_module(self, _module: ModuleSpec):
        """needs to reify parts of _module.spec and add them to _module

        Args:
            module (ModuleSpec): _description_

        Returns:
            _type_: _description_
        """
        if self._verbose is True:
            print(f"{self.__class__.__name__} exec_module type={type(_module)}", file=sys.stderr)
        return None

    def get_resource_reader(self, fullname: str) -> Union[ResourceReader,None]:
        """Required for Loaders that implement resource loading

        Args:
            fullname (str): full name of the resource

        Returns:
            ResourceReader: _description_
        """
        if self._verbose is True:
            print(f"get_resource_reader fullname={fullname}")
        return None

    def load_dataset(self) -> None:
        """loads a dataset
        """

    def is_package(self, name) -> bool:
        if self._verbose is True:
            print(f"{self.__class__.__name__} is_package name={name}", file=sys.stderr)
        return True       

    def load_plugin(self) -> bool:
        """loads the plugin specified in self.module.spec

        looks for explainers under explainer/explainers. These are yaml files.
        looks for plugins under explainer/plugins. 
        These are directories where explainer wheels have been installed

        """
        if self._verbose is True:
            print(f"{self.__class__.__name__} load_plugin full_path={self._full_path}", file=sys.stderr)
        plugin_name = re.sub('-', '_', self._module.spec.name)
        plugin_directory = os.path.dirname(self._full_path)
        plugin_path = os.path.join(plugin_directory, plugin_name)
        if self._verbose is True:
            print(f"{self.__class__.__name__} load_plugin plugin_path={plugin_path}", file=sys.stderr)        
        # if plugin_path exists then load the plugin
        if os.path.exists(plugin_path) and plugin_path not in sys.path:
            if self._verbose is True:
                print(f"{self.__class__.__name__} adding {plugin_path} to sys.path", file=sys.stderr)
            self._module.plugin_path = plugin_path
            sys.path.insert(0, plugin_path)
            return True
        return False

    def spec(self, loader: yaml.SafeLoader,
             node: yaml.nodes.MappingNode) -> ExplainerSpec:
        """Construct a ExplainerSpec"""
        if self._verbose is True:
            print(f"{self.__class__.__name__} spec", file=sys.stderr)
        return ExplainerSpec(**loader.construct_mapping(node))

    def get_yaml_loader(self) -> Type[yaml.FullLoader]:
        """Returns the yaml loader for the tag !ExplainerSpec

        Returns:
            Type[SafeLoader]: !ExplainerSpec
        """
        if self._verbose is True:
            print(f"{self.__class__.__name__} get_yaml_loader", file=sys.stderr)
        loader = yaml.FullLoader
        loader.add_constructor("!ExplainerSpec", self.spec)
        return loader

class ExplainerMetaPathFinder(MetaPathFinder):
    """ExplainerMetaPathFinder imports yaml files that define a
       explanation's model, data, entry_point and plugin path
    """
    def find_spec(self, fullname: str, path: Union[Sequence[str],None], _target: Union[ModuleType,None] = None) -> ExplainerModuleSpec:
        """Returns ExplainerLoader if the path is a yaml file that includes a !ExplainerSpec tag

        Args:
            fullname (str): package name
            path (str): path to package
            _target (ModuleType, optional): Defaults to None.

        Returns:
            ModuleSpec: _description_
        """
        #print(f"{self.__class__.__name__} find_spec fullname={fullname} path={path}", file=sys.stderr)
        mod_name = fullname.split('.')[-1]
        paths = path if path else [os.path.abspath(os.curdir)]
        for check_path in paths:
            full_path = os.path.join(check_path, mod_name + EXT_YAML)
            if os.path.exists(full_path):
                return spec_from_loader(check_path,  ExplainerLoader(full_path, verbose=False))  # type: ignore
        #print(f"{self.__class__.__name__} find_spec returning None", file=sys.stderr)
        return None  # type: ignore

sys.meta_path.insert(0, ExplainerMetaPathFinder())
