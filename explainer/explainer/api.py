"""
Explainer API
"""
import os
import sys
import re
import zipfile
from shutil import copy, move, rmtree
from subprocess import run, CalledProcessError, CompletedProcess, SubprocessError
from abc import ABC
from typing import Any, Callable, List, Tuple, Union

from . import ExplainerLoader, ExplainerModuleSpec, ExplainerPlugin, ExplainerSpec, EXT_YAML

explainers_folder = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "explainers"))


class Explainer(ABC):
    """Explainer API base class
    """
    def __init__(self, yamlpath:str=None):
        """Implements the Explainer API

        """
        self._explainer: Callable
        self._yamlpath = yamlpath

    def __call__(self, **kwargs):
        pass

    @property
    def explainer(self) -> Any:
        """Loads the explainer and returns it

        Returns:
            Any: any explainer for now
        """
        return self._explainer

    @property
    def list(self) -> List[str]:
        """Return the list of yaml files in the module explainer.explainers

        Returns:
            list: list of explainers
        """
        suffix = ".yaml"
        r_var: list = []
        for filename in os.listdir(explainers_folder):
            if filename.endswith(suffix):
                r_var.append(filename[0:-len(suffix)])
        r_var.sort()
        return r_var

    @property
    def update(self) -> bool:
        gbi = self.generate()
        if gbi is False:
            print(f"Error calling explainer generate", file=sys.stderr)
            return False
        gbi = self.build()
        if gbi is False:
            print(f"Error calling explainer build", file=sys.stderr)
            return False
        gbi = self.install(nowarnings=True)
        if gbi is False:
            print(f"Error calling explainer install", file=sys.stderr)
            return False
        return True

    def get_fullpath(self, yamlpath, addsuffix=True):
        suffix = ''
        if addsuffix is True:
            suffix = EXT_YAML
        if yamlpath is None:
            yamlpath = self._yamlpath
        if yamlpath is None:
            return None, None
        fullpath = os.path.abspath(os.path.join(explainers_folder, yamlpath+suffix))
        return fullpath, yamlpath

    def build(self, yamlpath:str=None) -> bool:
        """builds the plugin (wheel) and moves it to explainer/plugins

        Args:
            yamlpath (str): the yaml file path

        """
        def check_for_errors(cmd, sub_cmd, error) -> bool:
            if error is not None:
                print(f"{error}", file=sys.stderr)
                return True
            if sub_cmd is not None and sub_cmd.stderr is not None:
                print(f"{cmd}: {sub_cmd.stderr}", file=sys.stderr)
                return True
            return False

        fullpath, yamlpath = self.get_fullpath(yamlpath)
        module: Union[ExplainerModuleSpec, None] = None
        if os.path.exists(fullpath):
            module = ExplainerLoader(fullpath, verbose=False).module
            if module is not None:
                spec: ExplainerSpec = module.spec
                if spec is not None:
                    if spec.plugin is not None:
                        dirpath: str = os.path.join(
                          os.path.dirname(
                          os.path.dirname(fullpath)), "plugins")
                        parts: Tuple[str, str] = os.path.splitext(
                          os.path.basename(fullpath))
                        dirname: str = parts[0]
                        pluginname: str = re.sub('-', '_', spec.name)
                        pluginpath = os.path.join(dirpath, pluginname)
                        if os.path.exists(pluginpath):
                            os.chdir(pluginpath)
                            build_cmd: Union[CompletedProcess, None] = None
                            cmd = f"{sys.executable} setup.py bdist_wheel"
                            try:
                                build_cmd = run(cmd.split(), capture_output=True,
                                                text=True)
                                build_cmd.check_returncode()
                                whlname: str = spec.plugin
                                whlpath = os.path.join(pluginpath, "dist", whlname)
                                if os.path.exists(whlpath):
                                    targetpath = os.path.join(dirpath, whlname)
                                    mpath = move(whlpath, targetpath)
                                    if os.path.exists(mpath) is False:
                                      print(f"Could not move {whlpath} to {targetpath}.", file=sys.stderr)
                            except CalledProcessError as cpe:
                                return check_for_errors(cmd, build_cmd, cpe) is False
                            except SubprocessError as sube:
                                return check_for_errors(cmd, build_cmd, sube) is False
                            except ValueError as vae:
                                return check_for_errors(cmd, build_cmd, vae) is False
                        else:
                            print(f"{pluginpath}: does not exist, use the generate subcommand to create the plugin", file=sys.stderr)
                            return False
        else:
            print(f"{fullpath}: does not exist", file=sys.stderr)
            return False
        return True

    def extract(self, yamlpath:str=None) -> bool:
        """unpacks the plugin (wheel) to the directory explainer/plugins

        Args:
            yamlpath (str): the yaml file path

        """
        fullpath, yamlpath = self.get_fullpath(yamlpath)
        module: Union[ExplainerModuleSpec, None] = None
        if os.path.exists(fullpath):
            module = ExplainerLoader(fullpath, verbose=False).module
            if module is not None:
                spec: ExplainerSpec = module.spec
                if spec is not None:
                    if spec.plugin is not None:
                        dirpath: str = os.path.join(
                          os.path.dirname(
                            os.path.dirname(fullpath)), "plugins")
                        parts: Tuple[str, str] = os.path.splitext(
                          os.path.basename(fullpath))
                        dirname: str = parts[0]
                        whlname: str = spec.plugin
                        whlpath = os.path.join(dirpath, whlname)
                        pluginname: str = re.sub('-', '_', spec.name)
                        pluginpath = os.path.join(dirpath, pluginname)
                        pluginname: str = re.sub('-', '_', spec.name)
                        pyfile = f"{pluginname}.py"
                        pypath = os.path.join(pluginpath, pyfile)
                        if os.path.exists(whlpath):
                            self.generate(yamlpath)
                            archive = zipfile.ZipFile(whlpath)
                            for file in archive.namelist():
                                if file.startswith(pyfile):
                                    archive.extract(file, pluginpath)
                        else:
                            print(f"{targetpath}: exists, remove before installing {whlname}", file=sys.stderr)
                            return False
        else:
            print(f"{fullpath}: does not exit", file=sys.stderr)
            return False
        return True

    def generate(self, yamlpath:str=None) -> bool:
        """generates a template plugin directory under explainer/plugins

        Args:
            yamlpath (str): the yaml file path

        """
        fullpath, yamlpath = self.get_fullpath(yamlpath)
        module: Union[ExplainerModuleSpec, None] = None
        if os.path.exists(fullpath):
            module = ExplainerLoader(fullpath, verbose=False).module
            if module is not None:
                spec: ExplainerSpec = module.spec
                if spec is not None:
                    dirpath: str = os.path.join(os.path.dirname(os.path.dirname(fullpath)), "plugins")
                    parts: Tuple[str, str] = os.path.splitext(os.path.basename(fullpath))
                    dirname: str = parts[0]
                    dirpath = os.path.join(dirpath, dirname)
                    if os.path.exists(dirpath) is False:
                        os.mkdir(dirpath)
                    if os.path.exists(dirpath):
                        try:
                            explainer_plugin = ExplainerPlugin(spec)
                            plugin_dict = explainer_plugin.files
                            for plugin_file in plugin_dict:
                                fp = os.path.join(dirpath, plugin_file.name)
                                try:
                                    if os.path.exists(fp) and plugin_file.overwrite == False:
                                        continue
                                    with open(fp, 'w', encoding='utf-8') as f:
                                        fc = plugin_file.create()
                                        f.write(fc)
                                except ValueError as vae:
                                    print(f"could not generate {plugin_file.name}: {vae}", sys.stderr)
                                    return False
                        except ValueError as vae:
                            print(f"could not create explainer plugin generator: {vae}", sys.stderr)
                            return False
        else:
            print(f"{fullpath}: does not exist", file=sys.stderr)
            return False
        return True

    def import_from(self, yamlpath:str=None) -> Union[ExplainerModuleSpec, None]:
        """Import a plugin using a yaml file located under explainer/explainers

        If the yaml file specifies a name which is a directory under explainer/explainers then this is imported into 
        the current python environment and returned ExplainerModuleSpec overloads __getitem__ to return 
        any of the plugin's entry_points. 


        Args:
            yamlpath (str): path to the yaml file

        Returns:
            ExplainerModuleSpec: A ModuleSpec subclass
        """
        fullpath, yamlpath = self.get_fullpath(yamlpath)
        return ExplainerLoader(full_path, verbose=False).module

    def info(self, yamlpath:str=None) -> Union[ExplainerModuleSpec,None]:
        """shows information about the yaml file

        Args:
            yamlpath (str): path to the yaml file

        Returns:
            ExplainerModuleSpec: A ModuleSpec subclass
        """
        fullpath, yamlpath = self.get_fullpath(yamlpath)
        return ExplainerLoader(fullpath).module

    def install(self, yamlpath:str=None, nowarnings=False) -> Union[ExplainerModuleSpec, None]:
        """installs the plugin (wheel) under the directory explainer/explainers

        Args:
            yamlpath (str): the yaml file path

        """
        def check_for_errors(cmd, sub_cmd, error) -> None:
            if error is not None:
                print(f"{error}", file=sys.stderr)
            if sub_cmd is not None and sub_cmd.stderr is not None:
                print(f"{cmd}: {sub_cmd.stderr}", file=sys.stderr)

        fullpath, yamlpath = self.get_fullpath(yamlpath)
        module: Union[ExplainerModuleSpec, None] = None
        if os.path.exists(fullpath):
            module = ExplainerLoader(fullpath, verbose=False).module
            if module is not None:
                spec: ExplainerSpec = module.spec
                if spec is not None:
                    if spec.plugin is not None:
                        dirpath: str = os.path.join(
                          os.path.dirname(
                            os.path.dirname(fullpath)), "plugins")
                        parts: Tuple[str, str] = os.path.splitext(
                          os.path.basename(fullpath))
                        dirname: str = parts[0]
                        whlname: str = spec.plugin
                        whlpath = os.path.join(dirpath, whlname)
                        if os.path.exists(whlpath):
                            targetpath = os.path.join(explainers_folder, dirname)
                            if os.path.exists(targetpath) is True:
                                pyfile = os.path.join(dirpath, yamlpath, yamlpath+'.py')
                                if os.path.exists(pyfile) is True:
                                    if nowarnings == False:
                                        print(f"only copying {pyfile} to {targetpath}", file=sys.stderr)
                                    cpath = copy(pyfile, targetpath)
                                    if os.path.exists(cpath) is False:
                                      print(f"Could not copy {pyfile} to {targetpath}.", file=sys.stderr)
                            else:
                                install_cmd: Union[CompletedProcess, None] = None
                                cmd = f"{sys.executable} -m pip install {whlpath} --target {targetpath}"
                                try:
                                    install_cmd = run(cmd.split(), capture_output=True, text=True)
                                    install_cmd.check_returncode()
                                except CalledProcessError as cpe:
                                    check_for_errors(cmd, install_cmd, cpe)
                                except SubprocessError as sube:
                                    check_for_errors(cmd, install_cmd, sube)
                                except ValueError as vae:
                                    check_for_errors(cmd, install_cmd, vae)
                        else:
                            print(f"{whlpath}: does not exist, build the wheel by calling the subcommand build.",
                                  file=sys.stderr)
        else:
            print(f"{fullpath}: does not exit", file=sys.stderr)
        return module

    def load(self, yamlpath:str=None) -> Union[ExplainerModuleSpec, None]:
        """loads the plugin under the directory explainer/explainers

        Args:
            yamlpath (str): the yaml file path

        """
        fullpath, yamlpath = self.get_fullpath(yamlpath, addsuffix=True)
        module: Union[ExplainerModuleSpec, None] = None
        if os.path.exists(fullpath):
            module = ExplainerLoader(fullpath, verbose=False).module
        else:
            print(f"{fullpath}: does not exit", file=sys.stderr)
        return module

    def uninstall(self, yamlpath:str=None) -> bool:
        """uninstalls the plugin related directory under explainer/explainers

        Args:
            yamlpath (str): the yaml file path

        """
        fullpath, yamlpath = self.get_fullpath(yamlpath, addsuffix=False)
        if os.path.exists(fullpath):
            rmtree(fullpath, ignore_errors=True)
            if os.path.exists(fullpath):
                print(f"Could not remove {fullpath}", file=sys.stderr)
                return False
        else:
            print(f"{fullpath}: does not exist", file=sys.stderr)
            return False
        return True

    def unload(self, yamlpath:str=None) -> bool:
        """Unload the plugin using info from the yaml file to get the plugin path

        If the yaml file specifies a name which is a directory under explainer/explainers then 
        this function will unload the plugin from sys.path


        Args:
            yamlpath (str): path to the yaml file

        Returns:
            ExplainerModuleSpec: A ModuleSpec subclass
        """
        fullpath, yamlpath = self.get_fullpath(yamlpath, addsuffix=False)
        if os.path.exists(fullpath):
            if fullpath in sys.path:
                #print(f"Calling sys.path.remove", file=sys.stderr)
                sys.path.remove(fullpath)
                if fullpath in sys.path:
                    print(f"Could not remove {fullpath} from sys.path", file=sys.stderr)
                    return False
        else:
            print(f"{fullpath} does not exist", file=sys.stderr)
        return True

class ExplainerContextManager:
    """ExplainerContextManager
       
       Will load a plugin for the duration of the context and then unload it
       Example:

       from explainer.api import ExplainerContextManager as ec
           with ec('feature_attributions_explainer') as fe:
               pe = fe.partitionexplainer
               ??pe
    """
    def __init__(self, plugin_name, update=False):
        self.plugin_name = plugin_name
        self.explainer = Explainer(plugin_name)
        if update is True:
            self.explainer.update or print(f"Could not update explainer", file=sys.stderr)

    def __enter__(self):
        self.module_spec = self.explainer.load(self.plugin_name)
        if self.module_spec == None:
            print(f"Could not load {self.plugin_name}", file=sys.stderr)
        self.module_spec.entry_points
        return self.module_spec

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.explainer.unload(self.plugin_name) is not True:
            print(f"Could not unload {self.plugin_name}", file=sys.stderr)
        if exc_value is not None:
            # Handle error here...
            print(f"An exception occurred in your with block: {exc_type}", file=sys.stderr)
            print(f"Exception message: {exc_value}", file=sys.stderr)
            return True
        #print(f"contextmanager __exit__ sys.path={sys.path}", file=sys.stderr)
        return False
