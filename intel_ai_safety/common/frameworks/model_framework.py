from intel_ai_safety.common.constants import ModelFramework
import logging
from pydoc import locate


module_logger = logging.getLogger(__name__)

try:
    import torch.nn as nn
except ImportError:
    module_logger.debug('Could not import torch, required if using a PyTorch model')

MODEL_TYPE_NAMES = [
    "torch.nn.Module",
    "keras.engine.sequential.Sequential",
    "keras.engine.functional.Functional",
    ]


def is_tf_model(model):
    """Returns whether model is TF keras sequential or functional"""
    is_keras_sequential = str(type(model)).endswith("keras.engine.sequential.Sequential'>")
    is_keras_functional = str(type(model)).endswith("keras.engine.functional.Functional'>")
    is_keras_src_functional = str(type(model)).endswith("keras.src.engine.functional.Functional'>")
    is_keras_src_sequential = str(type(model)).endswith("keras.src.engine.sequential.Sequential'>")
    return is_keras_sequential | is_keras_functional | is_keras_src_functional | is_keras_src_sequential


def is_pt_model(model):
    """Returns whether model is PyTorch torch.nn.Module"""
    return isinstance(model, nn.Module)


def raise_unknown_model_error(model):
    raise ValueError(f"Model {type(model)} unsupported: please use model from {' ,'.join(MODEL_TYPE_NAMES)}")


def get_model_framework(model):
    """ Returns ModelFramework enum value corresponding to model.
    ModelFramework.TENSORFLOW

    Returns:
      ModelFramework: ModelFramework.TENSORFLOW or ModelFramework.PYTORCH
    
    Raises:
      ValueError: when model is not identified as TENSORFLOW or PYTORCH
    """
    if is_tf_model(model):
        return ModelFramework.TENSORFLOW
    elif is_pt_model(model):
        return ModelFramework.PYTORCH
    else:
        raise_unknown_model_error(model)


def is_torch_tensor(obj):
    """ Returns True when torch is installed and obj is a PyTorch tensor
    and False otherwise

    Returns:
      bool: True when obj is torch.Tensor False otherwise
    """
    TorchTensor = locate('torch.Tensor')
    if TorchTensor is not None:
        # Torch is installed and object is torch Tensor
        return isinstance(obj, TorchTensor)
    else:
        return False