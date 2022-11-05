from typing import Union
from numpy.typing import NDArray
from model_card_gen.datasets import TensorflowDataset, PytorchDataset

Array = NDArray
DatasetType = Union[TensorflowDataset, PytorchDataset]
