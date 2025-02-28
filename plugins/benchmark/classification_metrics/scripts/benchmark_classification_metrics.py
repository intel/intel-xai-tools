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
import habana_frameworks.torch.core as htcore
htcore.hpu_set_env()

import os
from pathlib import Path
import json
import pandas as pd
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

import evaluate
import torch
import argparse
from sklearn.metrics import precision_recall_curve, auc


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    return parser.parse_args()

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

def load_model(model_path):
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return (model, tokenizer)
    except:
        raise EnvironmentError("Please make sure that a valid model path is provided.")

def compute_metrics(all_preds, all_labels, all_probs):
    logger.info("loading accuracy metric")
    accuracy = evaluate.load("accuracy")
    logger.info("loading auroc metric")
    roc_auc = evaluate.load("roc_auc")
    logger.info("loading f1 metric")
    f1_metric = evaluate.load("f1")
    logger.info("loading precision metric")
    precision_metric = evaluate.load("precision")
    logger.info("loading recall metric")
    recall_metric = evaluate.load("recall")

    acc = accuracy.compute(predictions=all_preds, references=all_labels)
    logger.info(f"Accuracy: {acc}")
    auroc = roc_auc.compute(prediction_scores=all_probs, references=all_labels)
    logger.info(f"AUROC: {auroc}")
    f1 = f1_metric.compute(predictions=all_preds, references=all_labels)
    logger.info(f"F1: {f1}")
    precision = precision_metric.compute(predictions=all_preds, references=all_labels)
    logger.info(f"Precision: {precision}")
    recall = recall_metric.compute(predictions=all_preds, references=all_labels)
    logger.info(f"Recall: {recall}")

    false_positives = np.sum((np.array(all_preds) == 1) & (np.array(all_labels) == 0))
    logger.info(f"False Positives: {false_positives}")
    true_negatives = np.sum((np.array(all_preds) == 0) & (np.array(all_labels) == 0))
    logger.info(f"True Negatives: {true_negatives}")
    fpr = false_positives / (false_positives + true_negatives)
    logger.info(f"False Positive Rate: {fpr}")

    precision_temp, recall_temp, thresholds = precision_recall_curve(all_labels, all_preds)
    auc_precision_recall = auc(recall_temp, precision_temp)
    logger.info(f"AUC Precision Recall: {auc_precision_recall}")

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
    logger.info(f"Arguments: {args}")

    CHECKPOINT_NAME = os.path.basename(args.model_path)
    if args.results_path is None:
        WORKFLOW_DIR = Path(args.model_path).parent.absolute().parent.absolute()
    else:
        WORKFLOW_DIR = args.results_path
    TEST_RESULTS_PATH = os.path.join(WORKFLOW_DIR, "results", f"{CHECKPOINT_NAME}_{args.dataset_name}_accuracy")

    logger.info(f"Saving results in {TEST_RESULTS_PATH}")

    if not os.path.exists(TEST_RESULTS_PATH):
        os.makedirs(TEST_RESULTS_PATH)

    if args.dataset_name in ["tc"]:
        DATA_PATH = "hf://datasets/lmsys/toxic-chat/data/0124/toxic-chat_annotation_test.csv"
        df = pd.read_csv(DATA_PATH)
        test_dataset = Dataset.from_pandas(df)
    else:
        print(f"Support for dataset is coming soon...")
        exit(1)

    # load models
    model, tokenizer = load_model(args.model_path)

    # get the test dataset
    def preprocess_function(examples):
        return tokenizer(examples["user_input"], max_length=128)

    test_dataset = test_dataset.map(preprocess_function, batched=False, remove_columns=["conv_id", "user_input", "human_annotation", "jailbreaking", "model_output", "openai_moderation"])

    logger.info(f"Created toxic_chat test dataset.")

    collator = DataCollatorWithPadding(tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=64, collate_fn=collator)

    device = torch.device("hpu")
    logger.info(f"Using device: {device}")
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating", unit="batch"):
            # Move data to Habana device
            input_ids = torch.tensor(batch["input_ids"]).to(device)
            attention_mask = torch.tensor(batch["attention_mask"]).to(device)
            labels = torch.tensor(batch["toxicity"]).to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.nn.functional.softmax(outputs.logits)[:, 1]
            predictions = torch.argmax(outputs.logits, dim=-1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

    logger.info(f"Predictions: {all_preds[:20]}")
    logger.info(f"Labels: {all_labels[:20]}")
    logger.info(f"Probabilities: {all_probs[:20]}")

    results = compute_metrics(all_preds, all_labels, all_probs)

    json.dump(results, open(os.path.join(TEST_RESULTS_PATH, "metrics.json"), "w"))


if __name__ == "__main__":
    main()
