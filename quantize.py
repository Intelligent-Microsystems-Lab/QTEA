#!/usr/bin/env python3
"""Quantize a Llama-3 or Qwen3 model with QTEA and save the packed checkpoint.

    python quantize.py --model Qwen/Qwen3-8B-Base --output packed/qwen3-8b.pt
"""

import argparse
import time

import torch

from qtea.data import calibration_samples
from qtea.qtea import QTEAConfig
from qtea.model import DTYPES, load_model
from qtea.pack import save
from qtea.sequential import quantize_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="HuggingFace model id or local path")
    parser.add_argument("--output", required=True, help="where to write the packed checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16", choices=sorted(DTYPES))

    calibration = parser.add_argument_group("calibration")
    calibration.add_argument("--calibration-set", default="wikitext2", choices=["wikitext2", "c4"])
    calibration.add_argument("--nsamples", type=int, default=256)
    calibration.add_argument("--seqlen", type=int, default=2048)
    calibration.add_argument("--seed", type=int, default=0)

    method = parser.add_argument_group("QTEA")
    method.add_argument("--group-size", type=int, default=128, help="columns per quantization group")
    method.add_argument("--salient-ratio", type=float, default=0.05, help="fraction of columns with an FP8 residual")
    method.add_argument("--delta-coef", type=float, default=0.96, help="initial ternary dead zone")
    method.add_argument("--itf-iters", type=int, default=1, help="iterative ternary fitting rounds")
    method.add_argument("--rescale-iters", type=int, default=2,
                        help="column rescale refinement rounds; 0 pins v=1 and disables it")
    method.add_argument("--decay", type=float, default=1.0, help="GPTQ error decay lambda")
    method.add_argument("--damp", type=float, default=0.007, help="Hessian dampening")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    print(f"loading {args.model}")
    model, seqlen = load_model(args.model, args.dtype, args.seqlen)

    print(f"tokenizing {args.nsamples} calibration samples from {args.calibration_set}")
    dataloader = calibration_samples(
        args.calibration_set, args.model, nsamples=args.nsamples, seqlen=seqlen, seed=args.seed
    )

    config = QTEAConfig(
        group_size=args.group_size,
        salient_ratio=args.salient_ratio,
        delta_coef=args.delta_coef,
        itf_iters=args.itf_iters,
        rescale_iters=args.rescale_iters,
        decay=args.decay,
        damp=args.damp,
    )
    start = time.time()
    layers = quantize_model(model, dataloader, args.device, config)
    print(f"took {(time.time() - start) / 60:.1f} min")

    save(layers, args.model, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
