"""Defines the Explainer API
"""
import os
from abc import ABC
from typing import Any, Callable
from numpy.typing import ArrayLike
from . import ExplainerSpec, ExplainerModuleSpec

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

    def explainers(self) -> list:
        """Return the explainers available for the model

        Returns:
            list: list of explainers
        """
        r_var: list = []
        for filename in os.listdir(explainers_folder):
            if filename.endswith(".py"):
                r_var.append(filename[4:-3])
        r_var.sort()
        return r_var

    @property
    def explainer(self) -> Any:
        """Loads the explainer and returns it

        Returns:
            Any: any explainer for now
        """
        return self._explainer

    def import_from(self, _path: str) -> ExplainerModuleSpec:
        """import a yaml file using ExplainerLoader

        Args:
            _yamlpath (str): path to the yaml file

        Returns:
            ExplainerModuleSpec: A ModuleSpec subclass
        """
        return None


    def export_to(self, _yamlspec: ExplainerSpec) -> ExplainerModuleSpec:
        """create a yaml file using ExplainerSpec

        Args:
            _yamlpath (str): _description_

        Returns:
            ExplainerModuleSpec: _description_
        """
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
