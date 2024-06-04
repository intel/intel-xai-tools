import atheris
import json
import jsonschema
import sys

STR_BYTE_COUNT = 10000  # Desired byte count for fuzzed strings

default_path = "../model_card_gen"
sys.path.append(default_path)

with atheris.instrument_imports(include=["intel_ai_safety.*"]):
    from intel_ai_safety.model_card_gen.model_card_gen import ModelCardGen


def mutate_schema(fdp, json_data):
    """Recurses through a json object leaving keys and structures intact and
    randomly generating new data values of the proper type."""
    # TODO: Implement float, int, bool, and other types as needed
    if isinstance(json_data, str):
        return fdp.ConsumeUnicode(STR_BYTE_COUNT)
    elif isinstance(json_data, list):
        return [mutate_schema(fdp, json_data[i]) for i in range(len(json_data))]
    elif isinstance(json_data, dict):
        return {k: mutate_schema(fdp, v) for k, v in json_data.items()}
    else:
        return None


def TestOneInput(data):
    """The entry point for the fuzzer."""
    try:
        json_data = json.loads(data)
    except json.decoder.JSONDecodeError:
        print("Not valid json")
        return
    except UnicodeDecodeError:
        print("Not valid unicode")
        return

    fdp = atheris.FuzzedDataProvider(data)
    model_card_data = mutate_schema(fdp, json_data)
    try:
        mcg = ModelCardGen(data_sets={"test": ""}, model_card=model_card_data)
        if mcg.model_card:
            # TODO: Produces https://jira.devtools.intel.com/browse/AIZOO-3111
            mcg.build_model_card()  # Includes scaffold_assets() and export_format()
    except (ValueError, jsonschema.ValidationError):
        print("Doesn't match MC schema")
        return


if __name__ == "__main__":
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()
