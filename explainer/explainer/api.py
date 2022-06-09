"""Defines the Explainer API
"""
import os
from abc import ABC
from typing import Any, Callable
from numpy.typing import ArrayLike

explainers_folder = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "explainers"))


class Explanation:
    """_summary_
    """
    def __init__(self):
        pass

class Explainer(ABC):
    """Explainer API base class
    """

    def __init__(self,  model: Any = None) -> None:
        """takes any type of model or None

        Args:
            model (Any): any instance of any type of model
        """
        self._explainer: Callable = None
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
    def model(self):
        """returns the model

        Returns:
            Any: protected model
        """
        return self._model

    @property
    def explainer(self) -> Any:
        """Loads the explainer and returns it

        Returns:
            Any: any explainer for now
        """
        return self._explainer

    @property
    def expected_value(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self._explainer.expected_value

    def explain(self, _data: ArrayLike) -> Explanation:
        """takes _data and forwards to internal explainer

        Args:
            data (ArrayLike): any type of array

        Returns:
            Explanation: explainable objects
        """
        return self.explainer

    def visualize(self, *args: str, **kwargs: str) -> None:
        """_summary_
        """
