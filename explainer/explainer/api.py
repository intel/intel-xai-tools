"""Defines the Explainer API
"""
import os
import sys
import subprocess
import shutil
from abc import ABC
from typing import Any, Callable
from numpy.typing import ArrayLike

from . import ExplainerLoader, ExplainerModuleSpec, ExplainerSpec

explainers_folder = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "explainers"))

class Explainer(ABC):
    """Explainer API base class
    """

    def __init__(self,  model: Any = None) -> None:
        """takes any type of model or None

        Args:
            model (Any): any instance of any type of model
        """
        self._explainer: Callable
        self._model: Any = model

    def __call__(self, **kwargs):
        """allows arguments to be curried into a context

        Returns:
            _type_: _description_
        """
        if "model" in kwargs:
            self._model = kwargs["model"]

        return self

    @property
    def explainables(self) -> list[str]:
        """Return the explainable yaml files

        Returns:
            list: list of explainables
        """
        suffix = ".yaml"
        r_var: list = []
        for filename in os.listdir(explainers_folder):
            if filename.endswith(suffix):
                r_var.append(filename[0:-len(suffix)])
        r_var.sort()
        return r_var

    @property
    def explainer(self) -> Any:
        """Loads the explainer and returns it

        Returns:
            Any: any explainer for now
        """
        return self._explainer

    def import_from(self, yamlpath: str) -> ExplainerModuleSpec:
        """import a yaml file using ExplainerLoader

        import sys
        sys.path.insert(0, _archive)
        call __main__

        Args:
            yamlpath (str): path to the yaml file

        Returns:
            ExplainerModuleSpec: A ModuleSpec subclass
        """
        fullpath = os.path.abspath(yamlpath)
        module: ExplainerModuleSpec = None
        if os.path.exists(fullpath):
            el=ExplainerLoader(fullpath)
            module=el.create_module()
        return module

    def export_to(self, yamlpath: str) -> ExplainerModuleSpec:
        """create a yaml file under explainer/explainers along with
            artifacts specified in the yaml file

        Args:
            yamlpath (str): the yaml file path

        Returns:
            ExplainerModuleSpec: _description_
        """
        fullpath = os.path.abspath(yamlpath)
        module: ExplainerModuleSpec = None
        if os.path.exists(fullpath):
            el=ExplainerLoader(fullpath)
            module=el.create_module()
            spec=module.spec
            if hasattr(spec, "plugin"):
                if hasattr(spec, "dependencies"):
                    dirname=os.path.splitext(spec.plugin)[0]
                    dirpath=os.path.join(os.curdir,dirname)
                    os.mkdir(dirpath)
                    if os.path.exists(dirpath):
                        cmd = f"{sys.executable} -m pip install "
                        cmd += " ".join(str(x) for x in spec.dependencies)
                        cmd += f" --target {dirpath}"
                        print(cmd)
                        try:
                            subprocess.run(cmd.split(), check=True)
                        except subprocess.CalledProcessError as cpe:
                            print(f"process error {cpe}")
                        try:
                            shutil.make_archive(dirname, format="zip", root_dir=dirpath)
                            shutil.rmtree(dirpath, ignore_errors=True)
                        except ValueError as vae:
                            print(f"error {vae}")
        return module

    def explain(self, data: ArrayLike) -> None:
        """takes _data and forwards to internal explainer

        Args:
            data (ArrayLike): any type of array

        Returns:
            None
        """
        if self._explainer is not None:
            self._explainer(data)

    def visualize(self, *args: str, **kwargs: str) -> None:
        """_summary_
        """
