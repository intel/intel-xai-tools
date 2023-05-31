import logging

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
    return is_keras_sequential | is_keras_functional


def is_pt_model(model):
    """Returns whether model is PyTorch torch.nn.Module"""
    return isinstance(model, nn.Module)


def raise_unknown_model_error(model):
    raise ValueError(f"Model {type(model)} unsupported: please use model from {' ,'.join(MODEL_TYPE_NAMES)}")