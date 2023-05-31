#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
#

class ConfusionMatrix:
  """
    Confusion matrix object that holds the accuracy for each classification type in Cij, where
    C is the confusion matrix, i is a ground truth label and j is the model's label prediction.
    Confusion matrices are tool to help initially diagnose how well the model performs across
    all class labels. Checking these accuracies on unseen test data (real-world data) is a 
    first step in identifying possible sample bias.

    This class supports both binary and multi-class classification. Currently, the accuracies
    are normalized across each ground truth (row).

    Args:
      ground truth: 2-d array (n_samples, n_classes) of the ground truth, integer one-hot-encoded, 
        correct target values that correspond to the examples used in testing
      predictions: 2-d array (n_samples, n_classes) of the one-hot-encoded, predicted probabilities 
        that align with the ground truth array
      labels: 1-d array of strings that index the label names to the one-hot encodings

    Attributes:
      y_gt: 1-d array of integer ground truth labels
      y_pred: 1-d array of integer class prediction labels 
      labels: 1-d array of label names (strings) corresponding to the integer label indexes
      arr: 2-d array of float64s holding the confusion matrix
      df: Pandas DataFrame holding the confusion matrix with the associated label names
        for the indexes and column names
      report: A string summary of performances for all classes including precision, recall, f1-score,
        and support.

    Methods:
      visualize: Plot the confusion matrix on a heatmap

    Reference:
      https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
  """
  def __init__(self, groundtruth, predictions, labels):
    import sklearn
    from sklearn.metrics import confusion_matrix as cmx
    from sklearn.metrics import classification_report
    import pandas as pd
    pd.options.plotting.backend = "plotly"
    import numpy as np

    # ground truth
    self.y_gt = np.argmax(groundtruth, axis=1)
    # model predictions
    self.y_pred = np.argmax(predictions, axis=1)
    # label names
    self.labels = labels
    # array representation of the confusion matrix
    self.arr = cmx(self.y_gt, self.y_pred, normalize='true')
    # dataframe representation of the confusion matrix
    self.df = pd.DataFrame(self.arr, index = [i for i in self.labels],
                columns = [i for i in self.labels]) 
    # str representation of metrics containing precision, recall, f1 and acc for each class
    self.report = classification_report(self.y_gt, self.y_pred, target_names=self.labels)


  def visualize(self):
    import matplotlib.pyplot as plt
    import seaborn as sn
    plt.figure(figsize=(10,7))
    sn.heatmap(self.df, annot=True)


class Plotter:
  """
    Plotter object that calculates and holds the necessary values for various plots commonly used in 
    post-training analysis on test data. In particular, these plots utilize thesholding the predicted
    probabilities in order to quantify the skill of the model. They provide behavior of the classification
    rates dependent upon these probability thresholds and are useful in diagnosing dataset imbalance issues.

    This class accepts both binary and multi-class classification.

    Args:
      ground truth: 2-d array (n_samples, n_classes) of the ground truth, integer one-hot-encoded, 
        correct target values that correspond to the examples used in testing
      predictions: 2-d array (n_samples, n_classes) of the one-hot-encoded, predicted probabilities 
        that align with the ground truth array
      labels: 1-d array of strings that index the label names to the one-hot encodings

    Attributes:
      y_gt: 2-d array of one-hot-encoded integer ground truth labels
      y_pred: 2-d array of class predicted probabilities 
      labels: 1-d array of label names (strings) corresponding to the integer label indexes
      arr: 2-d array of float64s holding the confusion matrix
      precision: Python dict that holds a 1-d array of precisions at every threshold step for each class label
        where each key corresponds to the class label index. Used for PR curve 
      recall: Python dict that holds a 1-d array of recalls at every threshold step for each class label
        where each key corresponds to the class label index. Used for PR curve
      tpr: Python dict that holds a 1-d array of true-positive rates at every threshold step for each class label
        where each key corresponds to the class label index. Used for PR curve
      fpr: Python dict that holds a 1-d array of false-positive rates at every threshold step for each class label
        where each key corresponds to the class label index. Used for PR curve 

    Methods:
      pr_curve: plots the precision recall curve for all classes present in labels
      roc_curve: plots the receiver-operator characteristics curve for all classes present in labels

    Reference:
      https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
      https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
  """  
  def __init__(self,groundtruth,predictions,labels):
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import roc_curve as sk_roc_curve
    import pandas as pd
    import numpy as np
    pd.options.plotting.backend = "plotly"
    

    self.y_gt = np.array(groundtruth)
    self.y_pred = np.array(predictions)
    self.labels = labels
    self.precision, self.recall = dict(), dict()
    self.tpr, self.fpr = dict(), dict()


    for i in range(len(self.labels)):
      self.precision[i], self.recall[i], _ = precision_recall_curve(self.y_gt[:, i], self.y_pred[:, i])
      self.fpr[i], self.tpr[i], _ = sk_roc_curve(self.y_gt[:, i], self.y_pred[:, i])

  def pr_curve(self):
    import plotly.express as px
    fig = px.line(title='PR Curve').update_layout(yaxis_title='Precision', xaxis_title='Recall')
    for i in range(len(self.labels)):
      fig.add_scatter(x=self.recall[i], y=self.precision[i], name=self.labels[i])

    return fig

  def roc_curve(self):
    import plotly.express as px
    fig = px.line(title="ROC Curve").update_layout(yaxis_title='TPR', xaxis_title='FPR')
    for i in range(len(self.labels)):
      fig.add_scatter(x=self.fpr[i], y=self.tpr[i], name=self.labels[i])

    return fig


class PStats:
  def __init__(self, command):
    import cProfile
    import io
    from contextlib import redirect_stdout
    f = io.StringIO()
    with redirect_stdout(f):
      cProfile.run(command, sort='tottime')
    self.profile = f.getvalue().split('\n')

  @property
  def summary(self):
    import pstats
    import pandas as pd
    import numpy as np
    cols = ['function calls', 'time']
    summary_data = self.profile[:4]
#    return summary_data[0].strip()
    data = np.array(summary_data[0].split())[[0,-2]].reshape(1,2)
#    data = data.astype(float)
    data_ = pd.DataFrame(data)
    data_.columns=cols
    data_['function calls'].astype(int)
    data_['time'].astype(float)
#    def swap_columns(df, col1, col2):
#      col_list = list(df.columns)
#      x, y = col_list.index(col1), col_list.index(col2)
#      col_list[y], col_list[x] = col_list[x], col_list[y]
#      df = df[col_list]
#      return df
    return data_

  @property
  def report(self):
    import pstats
    import pandas as pd
    core_profile = self.profile[4:]
    cols = core_profile[0].split()
    n = len(cols[:-1])
    data = [_.split() for _ in core_profile[1:]]
    data = [_ if len(_)==n+1 else _[:n]+[" ".join(_[n+1:])] for _ in data]
    data_ = pd.DataFrame(data, columns=cols)
    return data_


def confusion_matrix(groundtruth,predictions,labels):
  """
    Generates instantiation of a ConfusionMatrix object that holds necessary information
    and metrics from the confusion matrix

    Args:
      groundtruth: 2-d array (n_samples, n_classes) of the ground truth, integer one-hot-encoded, 
        correct target values that correspond to the examples used in testing
      predictions: 2-d array (n_samples, n_classes) of the one-hot-encoded, predicted probabilities 
        that align with the groundtruth array
      labels: 1-d array of strings that index the label names to the one-hot encodings

    Returns:
      ConfusionMatrix: ConfusionMatrix object instantiation

    Reference:
      https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

    Example:
      >>> y_true = [[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]]
      >>> y_pred = [[0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]]
      >>> label_names = ['cat', 'dog', 'horse']
      >>> from explainer.explainers import metrics_explainer
      >>> cm = metrics_explainer['confusionmatrix'](y_true, y_pred, label_names)
      >>> print(cm.df)
                 cat  dog  horse
      cat        0.5  0.5        0.0
      dog        0.0  1.0        0.0
      horse  0.0  0.0        1.0

  """
  cm = ConfusionMatrix(groundtruth, predictions, labels) 
  return cm


def plot(groundtruth,predictions,labels):
  """
    Generates instantiation of a Plotter object that holds necessary information
    to plot the precision-recall curve and receiver-operator characteristics curve

    Args:
      groundtruth: 2-d array (n_samples, n_classes) of the ground truth, integer one-hot-encoded, correct target values 
        that correspond to the examples used in testing
      predictions: 2-d array (n_samples, n_classes) of the one-hot-encoded, predicted probabilities 
        that align with the groundtruth array
      labels: 1-d array of strings that index the label names to the one-hot encodings

    Returns:
      Plotter: Plotter object instantiation

    Reference:
      https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html 

    Example:
      >>> y_true = [[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]]
      >>> y_pred = [[.002, .09, .89], [.01, .7, .29], [.3, .67, .03], [.55, .4, .05], [.03, .86, .11]]
      >>> label_names = ['cat', 'dog', 'horse']
      >>> from explainer.explainers import metrics_explainer
      >>> plotter = metrics_explainer['plot'](y_true, y_pred, label_names)
      >>> plotter.recall
      {0: array([1. , 1. , 0.5, 0.5, 0.5, 0. ]), 1: array([1. , 1. , 1. , 0.5, 0.5, 0. ]), 2: array([1., 1., 1., 1., 1., 0.])}

  """
  plotter = Plotter(groundtruth,predictions,labels)
  return plotter


def pstats(command):
  """
     Executes cProfile.run and saves the results in the PStats class as panda DataFrames. 
     Two DataFrames are stored - A summary and a report which call PStats.summary and PStats.report respectively.

     Args:
       command: command to run

     Returns:
       PStats: this class provides summary and report methods to display the DataFrames

     Reference:
       https://docs.python.org/3/library/profile.html
  """
  return PStats(command)
