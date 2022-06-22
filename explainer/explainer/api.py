"""Defines the Explainer API
"""
import os
from abc import ABC
from typing import Any, Callable

import yaml
from numpy.typing import ArrayLike

from . import ExplainerModuleSpec, ExplainerSpec

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

    def import_from(self, _archive: str) -> ExplainerModuleSpec:
        """import a yaml file using ExplainerLoader

        import sys
        sys.path.insert(0, _archive)
        call __main__

        Args:
            _yamlpath (str): path to the yaml file

        Returns:
            ExplainerModuleSpec: A ModuleSpec subclass
        """
        return None


    def export_to(self, _yamlpath: str) -> ExplainerModuleSpec:
        """create a yaml file under explainer/explainers along with
            artifacts specified in the yaml file

        Args:
            yamlpath (str): the yaml file path

        Returns:
            ExplainerModuleSpec: _description_
        """
        #if os.path.exists(yamlpath):
        #    with open(yamlpath, mode="wt", encoding="utf-8") as file:
        #        yaml.dump(yamlspec, file)

        #import zipapp
        #import io
        #temp = io.BytesIO()
        #zipapp.create_archive('myapp.pyz', temp, '/usr/bin/python2')
        #with open('myapp.pyz', 'wb') as f:
        #    f.write(temp.getvalue())
        #
        # logic
        # create a directory 
        # create a __main__.py that imports the yaml
        # create a requirements.txt
        # run 'python -m pip install -r requirements.txt --target myapp"
        # run 'python -m zipapp myapp'

        return None

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
