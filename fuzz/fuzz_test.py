import atheris
import json
import sys

default_path = "../model_card_gen"
sys.path.append(default_path)

with atheris.instrument_imports(include=["intel_ai_safety.*"]):
    from intel_ai_safety.model_card_gen.model_card_gen import ModelCardGen

def TestOneInput(data):
    """The entry point for the fuzzer."""
    # Consume as unicode if the input is raw bytes
    if type(data) == bytes:
        fdp = atheris.FuzzedDataProvider(data)
        data = fdp.ConsumeUnicode(sys.maxsize)

    try:
        json_data = json.loads(data)
    except json.decoder.JSONDecodeError:
        print("Not valid json")
        return
    except UnicodeDecodeError:
        print("Not valid unicode")
        return
    except ValueError:
        print("Doesn't match MC schema")
        return

    mcg = ModelCardGen(data_sets={'test': ''}, model_card=json_data)
    print("Got a model card, its type is {}".format(type(mcg.model_card)))


if __name__ == '__main__':
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()
