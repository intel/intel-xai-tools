from model_card_gen.datasets import BaseDataset

class TensorflowDataset(BaseDataset):
    """
    Class wrapper for Tensorflow tfrecord
    """
    def __init__(self, dataset_path, name=""):
        super().__init__(dataset_path, name)
        self._framework = "tensorflow"
    
    @property
    def framework(self):
        """
        Returns the framework for dataset
        """
        return self._framework
