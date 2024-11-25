#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Intel Corporation
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
import os
from pathlib import Path
import json
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import evaluate
import torch
from torch.utils.data import Dataset
from torch.nn.functional import softmax
import argparse
from sklearn.metrics import precision_recall_curve, auc

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def parse_args():

    parser = argparse.ArgumentParser(
        prog='Test',
        description='Test model checkpoint with Jigsaw and ToxicChat datasets.',
        epilog='WIP'
    )

    parser.add_argument('-m', '--model-path', help='Path to the model checkpoint that \
                        will be tested.')
    parser.add_argument('-d', '--dataset-name', help='ToxicChat=\'tc\' or \
                        Jigsaw Unintended Bias=\'jigsaw\'')
    parser.add_argument('-r', '--results-path', help='Optional. Only set results path if you are \
                        testing model from hugging face hub.')
    parser.add_argument('-p', '--dataset-path', help='Required in case of Jigsaw dataset. Path of dataset file stored locally.')
    parser.add_argument('--device', type=str, default='hpu', help='Optional. Device Type: cpu or hpu. Will default to hpu.')
    parser.add_argument('-g_config','--gaudi_config_name', type=str, default='Habana/roberta-base', help='Optional. Name of the gaudi configuration. Will Default to Habana/roberta-base.')
    return parser.parse_args()


class BertoxDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def validate_metrics(predict_results):
    roc_auc = load_metric("roc_auc")
    n_samples = len(predict_results.label_ids)
    probabilities = softmax(torch.Tensor(predict_results.predictions))[:, 1]
    auroc = roc_auc.compute(prediction_scores=probabilities, references=predict_results.label_ids)
    preds = np.argmax(predict_results.predictions, axis=-1)

    n_correct = 0
    for i in range(n_samples):
        if preds[i] == predict_results.label_ids[i]:
            n_correct += 1
    accuracy = n_correct / n_samples

    print(f'My accuracy: {accuracy}.\nEvaluate accuracy: {predict_results.metrics["test_accuracy"]} ')
    print(f'My auroc: {auroc}.\nEvaluate accuracy: {predict_results.metrics["test_auroc"]} ')


def read_test_tc_split(csv_path):
    """
    Reads the test split for the ToxicChat dataset.
    """
    df = pd.read_csv(csv_path)
    texts = list(df.user_input)
    labels = list(df.toxicity)

    return texts, labels


def read_test_jigsaw_split(csv_path):
    """
    Reads the test split for the Jigsaw dataset.
    """
    try:
        df = pd.read_csv(csv_path, index_col=0)
        df = df[~df.comment_text.isna()]
        texts = list(df["comment_text"])
        labels = list(np.round(df["toxicity"]).astype(int))

        return texts, labels
    except:
        raise Exception(
            f"Error loading test dataset for Jigsaw Unintended Bias. Please ensure the CSV file path is correct and the file contains the required columns: 'comment_text' and 'toxicity'."
        )


def generate_datasets(test_texts, test_labels, tokenizer):
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)
    test_dataset = BertoxDataset(test_encodings, test_labels)

    return test_dataset


print("loading accuracy metric")
accuracy = evaluate.load("accuracy")
print("loading auroc metric")
roc_auc = evaluate.load("roc_auc")
print("loading f1 metric")
f1_metric = evaluate.load("f1")
print("loading precision metric")
precision_metric = evaluate.load("precision")
print("loading recall metric")
recall_metric = evaluate.load("recall")


def load_model(model_path):
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return (model, tokenizer)
    except:
        raise EnvironmentError("Please make sure that a valid model path is provided.")


def compute_metrics(eval_pred):

    logits, labels = eval_pred
    probabilities = softmax(torch.Tensor(logits))[:, 1]
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy.compute(predictions=predictions, references=labels)
    auroc = roc_auc.compute(prediction_scores=probabilities, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    precision = precision_metric.compute(predictions=predictions, references=labels)
    recall = recall_metric.compute(predictions=predictions, references=labels)

    false_positives = np.sum((predictions == 1) & (labels == 0))
    true_negatives = np.sum((predictions == 0) & (labels == 0))
    fpr = false_positives / (false_positives + true_negatives)

    precision_temp, recall_temp, thresholds = precision_recall_curve(labels, predictions)
    auc_precision_recall = auc(recall_temp, precision_temp)
    return {
        "accuracy": acc["accuracy"],
        "auroc": auroc["roc_auc"],
        "f1": f1["f1"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "fpr": fpr,
        "auprc": auc_precision_recall,
    }


def save_predictions(prediction_results, input_texts, results_path, CL):

    if CL:
        results_df = pd.DataFrame(
            {
                "input": input_texts,
                "target": prediction_results.label_ids,
                "toxic": prediction_results.predictions[:, 0],
                "not_toxic": prediction_results.predictions[:, 1],
            }
        )
    else:
        results_df = pd.DataFrame(
            {
                "input": input_texts,
                "target": prediction_results.label_ids,
                "not_toxic": prediction_results.predictions[:, 0],
                "toxic": prediction_results.predictions[:, 1],
            }
        )

    results_df.to_csv(results_path)
    return results_df


def main():
    args = parse_args()
    if args.device == "hpu":
        from optimum.habana import GaudiTrainer, GaudiTrainingArguments
    CL = False
    if "citizenlab" in args.model_path:
        CL = True

    CHECKPOINT_NAME = os.path.basename(args.model_path)
    if CL:
        WORKFLOW_DIR = Path(args.model_path).parent.absolute()
    elif args.results_path is None:
        WORKFLOW_DIR = Path(args.model_path).parent.absolute().parent.absolute()
    else:
        WORKFLOW_DIR = args.results_path

    TEST_RESULTS_PATH = os.path.join(WORKFLOW_DIR, "results", f"{CHECKPOINT_NAME}_{args.dataset_name}_accuracy")
    print(f"Saving results in {TEST_RESULTS_PATH}")

    if not os.path.exists(TEST_RESULTS_PATH):
        os.makedirs(TEST_RESULTS_PATH)

    if args.dataset_name in ["jigsaw", "tc"]:

        if args.dataset_name == "jigsaw":
            DATA_PATH = args.dataset_path
            if not DATA_PATH or not os.path.exists(DATA_PATH):
                raise FileNotFoundError(f"The specified dataset path does not exist or is not a directory.")
            DATA_PATH = Path(DATA_PATH)
            test_texts, test_labels = read_test_jigsaw_split(DATA_PATH)
        else:
            DATA_PATH = "hf://datasets/lmsys/toxic-chat/data/0124/toxic-chat_annotation_test.csv"
            test_texts, test_labels = read_test_tc_split(DATA_PATH)

    else:
        print(f"Support for dataset is coming soon...")
        exit(1)

    if CL:
        swap = {0: 1, 1: 0}
        test_labels = [swap[label] for label in test_labels]

    model, tokenizer = load_model(args.model_path)

    test_dataset = generate_datasets(test_texts, test_labels, tokenizer)
    training_args = GaudiTrainingArguments(
        output_dir=TEST_RESULTS_PATH,
        use_habana=True,
        use_lazy_mode=True,
        gaudi_config_name=args.g_config,
    )

    trainer = GaudiTrainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        eval_dataset=test_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,
    )

    results = trainer.predict(test_dataset)

    save_predictions(results, test_texts, os.path.join(TEST_RESULTS_PATH, "predictions.csv"), CL)
    json.dump(results.metrics, open(os.path.join(TEST_RESULTS_PATH, "metrics.json"), "w"))


if __name__ == "__main__":
    main()
