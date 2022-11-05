from model_card_gen.model_card_gen import ModelCardGen
from model_card_gen.tests.torch_model import get_data, get_trained_model
from model_card_gen.datasets import PytorchDataset
from google.protobuf import text_format
import tensorflow_model_analysis as tfma
import os


def test_end_to_end():
    """ Build a pytorch model card from a trained model
    """

    adult_dataset, feature_names = get_data()
    _model_path = get_trained_model(adult_dataset, feature_names)
    _data_sets ={'train': PytorchDataset(adult_dataset, feature_names=feature_names)}

    eval_config = text_format.Parse("""
    model_specs {
    label_key: 'label'
    prediction_key: 'prediction'
    }
    metrics_specs {
        metrics {class_name: "BinaryAccuracy"}
        metrics {class_name: "AUC"}
        metrics {class_name: "ConfusionMatrixPlot"}
        metrics {
        class_name: "FairnessIndicators"
        }
    }
    slicing_specs {}
    slicing_specs {
            feature_keys: 'sex_Female'
    }
    options {
        include_default_metrics { value: false }
    }
    """, tfma.EvalConfig())

    mcg = ModelCardGen.generate(data_sets=_data_sets, model_path=_model_path, eval_config=eval_config)
    # clean up pytorch model file
    os.remove(_model_path)
    assert mcg
