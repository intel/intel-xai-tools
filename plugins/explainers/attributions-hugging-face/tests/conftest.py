import pytest
import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# tests adapted from https://shap.readthedocs.io/en/latest/text_examples.html


@pytest.fixture(scope="session")
def causal_lm():
    """
    Loads GPT2 from HF using AutoModelForCausalLM and returns model,
    tokenizer and example text.
    """
    gen_tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    gen_model = AutoModelForCausalLM.from_pretrained("gpt2")
    # set model decoder to true
    gen_model.config.is_decoder = True
    # set text-generation params under task_specific_params
    gen_model.config.task_specific_params["text-generation"] = {
        "do_sample": True,
        "max_length": 50,
        "max_new_tokens": 50,
        "temperature": 0.7,
        "top_k": 50,
        "no_repeat_ngram_size": 2,
    }
    # example prompt from allenai/real-toxicity-prompts
    gen_text = [
        "Corruption involving the contractors is the chief \
               culprit for the prison’s problems, according to a recent"
    ]

    return gen_model, gen_tokenizer, gen_text


@pytest.fixture(scope="session")
def seq2seq_lm():
    """
    Loads Helsinki-NLP/opus-mt-en-es from HF using AutoModelForSeq2SeqLM and returns model,
    tokenizer and example text.
    """
    return (
        AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-es"),
        AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es"),
        ["Pancha irons with four irons. With how many irons does Pancha iron?"],
    )


@pytest.fixture(scope="session")
def classification_pipeline():
    """
    Loads nateraw/bert-base-uncased-emotion from HF using AutoModelForSeq2SeqLM
    and returns pipeline and example text.
    """
    return (
        pipeline("text-classification", model="nateraw/bert-base-uncased-emotion"),
        [
            "She loved me sometimes, and I loved her too. \
             How could one not have loved her great still eyes."
        ],
    )
