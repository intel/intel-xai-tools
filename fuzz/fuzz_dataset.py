import atheris
import numpy
import random
import sys

MIN_DATA_LENGTH = 1  # Minimum length of dataset
MAX_DATA_LENGTH = 1000  # Maximum length of dataset

default_path = "../plugins/model_card_gen/generators/tfma/"
sys.path.append(default_path)

with atheris.instrument_imports(include=["intel_ai_safety.*"]):
    from intel_ai_safety.model_card_gen.datasets.torch_datasets import PytorchNumpyDataset


def TestOneInput(data):
    """The entry point for the fuzzer."""
    fdp = atheris.FuzzedDataProvider(data)

    # Create input and target numpy arrays of random but equal length
    # Label values will be integers between [0, 10]
    dataset_length = random.randint(MIN_DATA_LENGTH, MAX_DATA_LENGTH)
    input_array = numpy.array(fdp.ConsumeRegularFloatList(dataset_length))
    target_array = numpy.array(fdp.ConsumeIntListInRange(dataset_length, 0, 10))

    dataset = PytorchNumpyDataset(input_array=input_array, target_array=target_array)
    assert len(dataset.dataset) == dataset_length


if __name__ == "__main__":
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()
