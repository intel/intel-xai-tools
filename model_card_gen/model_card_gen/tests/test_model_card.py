import pytest
import os
import pkgutil
import json
from model_card_gen.model_card_gen import ModelCardGen
from model_card_gen.validation import (_LATEST_SCHEMA_VERSION, 
                                             _SCHEMA_FILE_NAME,
                                             _find_json_schema,
                                             validate_json_schema)
PACKAGE = 'model_card_gen'
JSON_FILES = ['docs/examples/json/model_card_example.json', 'docs/examples/json/model_card_compas.json']
MODEL_CARD_STRS = [pkgutil.get_data(PACKAGE, json_file) for json_file in JSON_FILES]
MODEL_CARD_JSONS = [json.loads(json_str) for json_str in MODEL_CARD_STRS]

@pytest.mark.parametrize("test_json", MODEL_CARD_JSONS)
def test_init(test_json):
    """Test ModelCardGen initialization
    """
    mcg = ModelCardGen(test_json)
    assert mcg.model_card

@pytest.mark.parametrize("test_json", MODEL_CARD_JSONS)
def test_read_json(test_json):
    """Test ModelCardGen._read_json method
    """
    mcg = ModelCardGen(model_card=test_json)
    assert mcg.model_card == ModelCardGen._read_json(test_json)

@pytest.mark.parametrize("test_json", MODEL_CARD_JSONS)
def test_validate_json(test_json):
    """Test JSON validates
    """
    assert validate_json_schema(test_json) == _find_json_schema()

@pytest.mark.parametrize("test_json", MODEL_CARD_JSONS)
def test_schemas(test_json):
    """Test JSON schema loads
    """
    schema_file = os.path.join('schema', 'v' + _LATEST_SCHEMA_VERSION, _SCHEMA_FILE_NAME)
    json_file = pkgutil.get_data('model_card_gen', schema_file)
    schema = json.loads(json_file)
    assert schema == _find_json_schema(_LATEST_SCHEMA_VERSION)
