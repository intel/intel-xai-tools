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
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import time

def parse_args():

    parser = argparse.ArgumentParser(
        prog='Test',
        description='Test model checkpoint with Jigsaw and ToxicChat datasets.',
        epilog='WIP'
    )

    parser.add_argument('-m', '--model-path', help='Path to the model checkpoint that \
                        will be tested.')
    parser.add_argument('-d', '--dataset-name', help='BeaverTails=\'bt\' \
                        or Jigsaw Unintended Bias=\'jigsaw\' \
                        or OpenAI Moderation=\'mod\' or SurgeAI Toxicity=\'surgetox\' \
                        or ToxicChat=\'tc\' or ToxiGen=\'tgen\' \
                        or XSTest=\'xst\'')
    parser.add_argument('-r', '--results-path', help='Optional. Only set results path if you are \
                        testing model from hugging face hub.')
    parser.add_argument('-p', '--dataset-path', help='Required in case of Jigsaw, SurgeAI Toxicity and OpenAI Moderation. Path of dataset file stored locally.')
    parser.add_argument('--device', type=str, default='hpu', help='Optional. Device Type: cpu or hpu. Will default to hpu.')
    parser.add_argument('--batch_size', type=int, default=128, help='Optional. Batch size for processing data. Defaults to 128.')
    return parser.parse_args()

def read_test_split_tc(csv_path):
    """
    Reads the test split for the ToxicChat dataset.
    """
    df = pd.read_csv(csv_path)
    texts = list(df.user_input)
    labels = list(df.toxicity) 

    return texts, labels

def read_test_split_bt(csv_path): 
    """
    Reads the test split for the Beaver Tails dataset.
    """
    df = pd.read_json(csv_path, lines=True)
    texts = list(df["prompt"])
    labels = list(df["is_safe"].astype(int))

    return texts, labels

def read_test_split_xs(csv_path):
    """
    Reads the test split for the XSTest dataset.
    """
    df = pd.read_parquet(csv_path)
    texts = list(df["prompt"])
    label_map = {"safe":0, "unsafe":1}
    labels = [label_map[label] for label in list(df["label"])]

    return texts, labels

def read_test_split_tg(csv_path):
    """
    Reads the test split for the Toxigen dataset.
    """
    df = pd.read_parquet(csv_path)
    texts = list(df["prompt"])
    labels = list(df["prompt_label"])
    
    return texts, labels

def read_test_split_openaimod(json_path):
    """
    Reads the test split for the OpeanAI Mod dataset.
    """
    try:
        df = pd.read_json(json_path, lines=True)
        texts = list(df["prompt"])
        columns_to_check = ['S', 'H', 'V', 'HR', 'SH', 'S3', 'H2', 'V2']
        labels = list(df[columns_to_check].max(axis=1).astype(int))
        
        return texts, labels
    except:
        raise Exception(f"Error loading test dataset for SurgeAI Toxicity. Please ensure the CSV file path is correct.")
    
def read_test_split_surge(csv_path):
    """
    Reads the test split for the SurgeAI Toxicity dataset.
    """
    try:
        df = pd.read_csv(csv_path)
        texts = list(df["text"])
        label_map = {"Not Toxic":0, "Toxic":1}
        labels = [label_map[label] for label in list(df["is_toxic"])]
        
        return texts, labels
    except:
        raise Exception(f"Error loading test dataset for SurgeAI Toxicity. Please ensure the CSV file path is correct "
                        "and the file contains the required columns: 'text' and 'is_toxic'.")

def read_test_split_jigsaw(csv_path): 
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
        raise Exception(f"Error loading test dataset for Jigsaw Unintended Bias. Please ensure the CSV file path is correct "
            "and the file contains the required columns: 'comment_text' and 'toxicity'.")

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
    
    true_negatives, false_positives, false_negatives, true_positives = confusion_matrix(all_labels, all_preds).ravel()
    logger.info(f"True Negatives: {true_negatives}")
    logger.info(f"False Positives: {false_positives}")
    logger.info(f"False Negatives: {false_negatives}")
    logger.info(f"True Positives: {true_positives}")

    tnr, fpr, fnr, tpr = confusion_matrix(all_labels, all_preds, normalize='true').ravel()
    logger.info(f"True Positive Rate: {tpr}")
    logger.info(f"False Positive Rate: {fpr}")
    logger.info(f"True Negative Rate: {tnr}")
    logger.info(f"False Negative Rate: {fnr}")

    precision_prob, recall_prob, thresholds = precision_recall_curve(all_labels, all_probs)
    auc_precision_recall = auc(recall_prob, precision_prob)
    logger.info(f"AUC Precision Recall: {auc_precision_recall}")

    return {
        "accuracy": acc["accuracy"],
        "auroc": auroc["roc_auc"],
        "f1": f1["f1"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "true_positive_rate": tpr,
        "false_positive_rate": fpr,
        "true_negative_rate": tnr,
        "false_negative_rate": fnr,
        "auprc": auc_precision_recall,
        "false_positives": int(false_positives),
        "true_negatives": int(true_negatives),
        "true_positives": int(true_positives),
        "false_negatives": int(false_negatives),
    }

def save_predictions(prediction_results, labels, input_texts, results_path, CL):
    
    # one-hot encode preds
    n_values = np.max(np.array(prediction_results)) + 1
    preds = np.eye(n_values)[prediction_results]

    if CL:
        results_df = pd.DataFrame(
            {
                "input": input_texts,
                "target": labels,
                "toxic": preds[:, 0],
                "not_toxic": preds[:, 1],
            }
        )
    else:
        results_df = pd.DataFrame(
            {
                "input": input_texts,
                "target": labels,
                "not_toxic": preds[:, 0],
                "toxic": preds[:, 1],
            }
        )

    results_df.to_csv(results_path)
    return results_df


def main():
    args = parse_args()
    logger.info(f"Arguments: {args}")
    batch_size = args.batch_size
    
    read_test_split = {
    'tc': ('hf://datasets/lmsys/toxic-chat/data/0124/toxic-chat_annotation_test.csv', read_test_split_tc),
    'xst': ('hf://datasets/walledai/XSTest/data/train-00000-of-00001.parquet', read_test_split_xs),
    'bt': ('hf://datasets/PKU-Alignment/BeaverTails/round0/330k/test.jsonl.xz', read_test_split_bt),
    'tgen': ('hf://datasets/toxigen/toxigen-data/train/train-00000-of-00001.parquet', read_test_split_tg),
    'mod': (args.dataset_path, read_test_split_openaimod),
    'surgetox': (args.dataset_path, read_test_split_surge),
    'jigsaw': (args.dataset_path, read_test_split_jigsaw)
    }
    
    if args.device == "hpu":
        import habana_frameworks.torch.core as htcore
        htcore.hpu_set_env()
        logger.info("Habana environment initialized.")
        device = torch.device("hpu")
        logger.info(f"Using device: {device}")
    elif args.device == "cpu":
        device = torch.device("cpu")
        logger.info(f"Using device: {device}")

    CHECKPOINT_NAME = os.path.basename(args.model_path)
    if args.results_path is None:
        WORKFLOW_DIR = Path(args.model_path).parent.absolute().parent.absolute()
    else:
        WORKFLOW_DIR = args.results_path
    TEST_RESULTS_PATH = os.path.join(WORKFLOW_DIR, "results", f"{CHECKPOINT_NAME}_{args.dataset_name}_accuracy")

    logger.info(f"Saving results in {TEST_RESULTS_PATH}")

    if not os.path.exists(TEST_RESULTS_PATH):
        os.makedirs(TEST_RESULTS_PATH)
        
    if args.dataset_name in read_test_split:
        DATA_PATH, test_split_fn = read_test_split[args.dataset_name]
        
        if args.dataset_name in ['mod', 'surgetox', 'jigsaw']:
            if not DATA_PATH or not os.path.exists(DATA_PATH):
                raise FileNotFoundError(f"The specified dataset path does not exist or is not a directory.")
            DATA_PATH = Path(DATA_PATH)
        
        test_texts, test_labels = test_split_fn(DATA_PATH)
    else:
        print(f"Support for dataset is coming soon...")
        exit(1)

    # load models
    model, tokenizer = load_model(args.model_path)

    # get the test dataset
    def preprocess_function(examples):
        return tokenizer(examples["user_input"], max_length=128)

    test_dataset = Dataset.from_pandas(pd.DataFrame({"user_input": test_texts, "toxicity": test_labels}))
    test_dataset = test_dataset.map(preprocess_function, batched=False, remove_columns=["user_input"])
    logger.info(f"Created {args.dataset_name} test dataset.")

    collator = DataCollatorWithPadding(tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collator)

    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    start = time.perf_counter()

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
    end = time.perf_counter()

    logger.info(f"Time taken to evaluate: {end - start} seconds")
    logger.info(f"Predictions: {all_preds[:20]}")
    logger.info(f"Labels: {all_labels[:20]}")
    logger.info(f"Probabilities: {all_probs[:20]}")

    results = compute_metrics(all_preds, all_labels, all_probs)

    json.dump(results, open(os.path.join(TEST_RESULTS_PATH, "metrics.json"), "w"))
    save_predictions(all_preds, all_labels, test_texts, os.path.join(TEST_RESULTS_PATH, "predictions.csv"), False)


if __name__ == "__main__":
    main()
