import pandas as pd
import tensorflow_model_analysis as tfma
from typing import Text, Union
from model_card_gen.utils.types import DatasetType
from model_card_gen.analyze.analyzer import ModelAnalyzer

class DFAnalyzer(ModelAnalyzer):
    def __init__(self,
                 eval_config: Union[tfma.EvalConfig, Text] = None,
                 dataset: pd.DataFrame = None):
        """Start TFMA analysis on Pandas DataFrame

        Args:
            raw_data (pd.DataFrame): dataframe containing prediciton values and ground truth
            eval_config (tfma.EvalConfig or str): representing proto file path
        """
        super().__init__(eval_config, dataset)
    
    @classmethod
    def analyze(cls,
                eval_config: Union[tfma.EvalConfig, Text] = None,
                dataset:  DatasetType = None,):
        """Class Factory to start TFMA analysis
        Args:
            model_path (str) : path to model
            eval_config (tfma.EvalConfig or str): representing proto file path
            data (str or pd.DataFrame): string ot tfrecord or raw dataframe containing
                prediction values and  ground truth

        Raises:
            TypeError: when eval_config is not of type tfma.EvalConfig or str
            TypeError: when data argument is not of type pd.DataFrame or str

        Returns:
            tfma.EvalResults()

        Example:
            >>> from model_card_gen.analyzer import DFAnalyzer
            >>> DFAnalyzer.analyze(
                model_path='compas/model',
                data='compas/eval.tfrecord',
                eval_config='compas/eval_config.proto')
        """
        self = cls(eval_config, dataset)
        self.run_analysis()
        return self.get_analysis()

    def run_analysis(self):
        self.eval_result = tfma.analyze_raw_data(data=self.dataset,
                                                 eval_config=self.eval_config)
        return self.eval_result