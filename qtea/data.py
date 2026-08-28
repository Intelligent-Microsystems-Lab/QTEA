"""Calibration and evaluation data (WikiText-2 and C4)."""

import random

from datasets import load_dataset
from transformers import AutoTokenizer


def get_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name, use_fast=False)


def _wikitext2(tokenizer, split):
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    separator = " " if split == "train" else "\n\n"
    return tokenizer(separator.join(data["text"]), return_tensors="pt").input_ids


def _c4_train(tokenizer, nsamples, seqlen):
    data = load_dataset(
        "allenai/c4", data_files="en/c4-train.00000-of-01024.json.gz", split="train"
    )
    samples = []
    while len(samples) < nsamples:
        document = tokenizer(data[random.randint(0, len(data) - 1)]["text"], return_tensors="pt").input_ids
        if document.shape[1] <= seqlen:
            continue
        start = random.randint(0, document.shape[1] - seqlen - 1)
        samples.append(document[:, start : start + seqlen])
    return samples


def _c4_validation(tokenizer, seqlen):
    data = load_dataset(
        "allenai/c4", data_files="en/c4-validation.00000-of-00008.json.gz", split="train"
    )
    tokens = tokenizer(" ".join(data[:1100]["text"]), return_tensors="pt").input_ids
    return tokens[:, : 256 * seqlen]


def calibration_samples(dataset, model_name, nsamples=256, seqlen=2048, seed=0):
    """`nsamples` random windows of `seqlen` tokens, each shaped (1, seqlen)."""
    random.seed(seed)
    tokenizer = get_tokenizer(model_name)

    if dataset == "c4":
        return _c4_train(tokenizer, nsamples, seqlen)

    tokens = _wikitext2(tokenizer, "train")
    starts = [random.randint(0, tokens.shape[1] - seqlen - 1) for _ in range(nsamples)]
    return [tokens[:, start : start + seqlen] for start in starts]


def evaluation_tokens(dataset, model_name, seqlen=2048):
    """The full held-out token stream used for perplexity."""
    tokenizer = get_tokenizer(model_name)
    if dataset == "c4":
        return _c4_validation(tokenizer, seqlen)
    if dataset == "wikitext2":
        return _wikitext2(tokenizer, "test")
    raise ValueError(f"unknown dataset {dataset!r}")
