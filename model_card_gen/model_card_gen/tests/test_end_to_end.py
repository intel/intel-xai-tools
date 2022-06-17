import pytest
import tempfile
import tensorflow as tf
from model_card_gen.tests.model import build_and_train_model, train_tf_file, validate_tf_file
from model_card_gen.model_card_gen import ModelCardGen
import tensorflow_model_analysis as tfma
from google.protobuf import text_format


def test_end_to_end():
    """ Build a model card from a trained model
    """
    tfma_export_dir = build_and_train_model()
    _model_path = tfma_export_dir
    _data_paths = {'eval': validate_tf_file,
                'train': train_tf_file}

    eval_config = text_format.Parse("""
    model_specs {
        signature_name: "eval"
    }
    
    metrics_specs {
        metrics { class_name: "BinaryAccuracy" }
        metrics { class_name: "Precision" }
        metrics { class_name: "Recall" }
        metrics { class_name: "ConfusionMatrixPlot" }
        metrics { class_name: "FairnessIndicators" }
    }

    slicing_specs {}  # overall slice
    slicing_specs {
        feature_keys: ["gender"]
    }
    """, tfma.EvalConfig())

    mcg = ModelCardGen.generate(_data_paths, _model_path, eval_config)
    assert mcg