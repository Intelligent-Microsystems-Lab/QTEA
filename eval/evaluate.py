#!/usr/bin/env python3
"""Evaluate a QTEA checkpoint: WikiText-2 / C4 perplexity and zero-shot accuracy.

    python eval/evaluate.py --model Qwen/Qwen3-8B-Base --checkpoint packed/qwen3-8b.pt

Omit --checkpoint to measure the unquantized FP16 baseline.
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perplexity import perplexity
from qtea.data import evaluation_tokens
from qtea.model import DTYPES, load_model

ZERO_SHOT_TASKS = ["piqa", "arc_easy", "arc_challenge", "hellaswag", "winogrande", "openbookqa", "boolq"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="HuggingFace model id or local path")
    parser.add_argument("--checkpoint", help="packed checkpoint from quantize.py; omit for FP16")
    parser.add_argument("--backend", default="dequant", choices=["dequant", "lut"],
                        help="dequant: dense FP16 weights; lut: custom CUDA kernel")
    parser.add_argument("--ppl", default="wikitext2,c4", help="comma separated, or empty to skip")
    parser.add_argument("--tasks", default=",".join(ZERO_SHOT_TASKS), help="comma separated, or empty to skip")
    parser.add_argument("--limit", type=int, help="evaluate only N documents per task (smoke test)")
    parser.add_argument("--batch-size", default="8", help="zero-shot batch size, or 'auto'")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--dtype", default="float16", choices=sorted(DTYPES))
    parser.add_argument("--output", help="write the results to this JSON file")
    return parser.parse_args()


def split(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def run_zero_shot(model, tokenizer, tasks, device, limit, batch_size):
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    model.to(device)
    results = lm_eval.simple_evaluate(
        model=HFLM(pretrained=model, tokenizer=tokenizer, device=device, batch_size=batch_size),
        tasks=tasks,
        limit=limit,
    )["results"]

    accuracy = {}
    for task, metrics in results.items():
        accuracy[task] = {key.split(",")[0]: value for key, value in metrics.items()
                          if key.startswith(("acc,", "acc_norm,"))}
    model.cpu()
    torch.cuda.empty_cache()
    return accuracy


def main():
    args = parse_args()
    model, seqlen = load_model(args.model, args.dtype, args.seqlen)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.checkpoint:
        from packed_linear import install_checkpoint
        install_checkpoint(model, args.checkpoint, backend=args.backend)

    results = {"model": args.model, "checkpoint": args.checkpoint, "backend": args.backend}

    for dataset in split(args.ppl):
        tokens = evaluation_tokens(dataset, args.model, seqlen=seqlen)
        results[f"ppl/{dataset}"] = perplexity(model, tokens, args.device, seqlen)
        print(f"{dataset} perplexity: {results[f'ppl/{dataset}']:.4f}")

    tasks = split(args.tasks)
    if tasks:
        batch_size = 1 if args.backend == "lut" else args.batch_size
        accuracy = run_zero_shot(model, tokenizer, tasks, args.device, args.limit, batch_size)
        results["zero_shot"] = accuracy
        # Table 1 of the paper reports plain accuracy, so that is what we average.
        headline = []
        for task in tasks:
            score = accuracy.get(task, {}).get("acc")
            if score is not None:
                headline.append(score)
                print(f"{task}: {100 * score:.2f}")
        if headline:
            results["zero_shot_average"] = sum(headline) / len(headline)
            print(f"average: {100 * results['zero_shot_average']:.2f}")

    print(json.dumps(results, indent=2))
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
