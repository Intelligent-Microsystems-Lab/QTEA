#!/usr/bin/env python3
"""Check the LUT CUDA kernel of a packed checkpoint against dense reference math.

    python eval/check_kernel.py --checkpoint packed/qwen3-8b.pt

Reports the deviation from a dense FP16 matmul with the same weights, which
should sit at the FP16 accumulation floor (~1e-3 relative).

The per-call times next to it are a raw microbenchmark of one projection with a
single-token input.  They are dominated by launch overhead and say little about
end-to-end serving: the latency numbers in the paper come from full-model
generation under a captured CUDA graph, where the kernel's smaller weight
footprint is what pays off.
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtea.pack import load, unpack_layer
from ternary_kernel import TernaryGEMV


def benchmark(function, warmup=20, iterations=200):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        function()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iterations * 1e6  # microseconds


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", type=int, default=4, help="how many projections to check")
    args = parser.parse_args()

    layers, model_name = load(args.checkpoint)
    print(f"{model_name}: {len(layers)} packed projections\n")
    print(f"{'projection':<34}{'shape':>16}{'rel err':>12}{'dense us':>12}{'lut us':>10}")

    torch.manual_seed(0)
    for name, layer in list(layers.items())[: args.layers]:
        gemv = TernaryGEMV(layer).to(args.device)
        dense = unpack_layer(layer).to(args.device, torch.float16)
        x = torch.randn(1, int(layer["cols"]), device=args.device, dtype=torch.float16)

        with torch.no_grad():
            error = (gemv(x).float() - (x @ dense.t()).float()).abs().max().item()
            reference = (x.float() @ dense.t().float()).abs().max().item()
            lut_us = benchmark(lambda: gemv(x))
            dense_us = benchmark(lambda: x @ dense.t())

        shape = f"{layer['rows']}x{layer['cols']}"
        print(f"{name:<34}{shape:>16}{error / reference:>12.2e}{dense_us:>12.1f}{lut_us:>10.1f}")
        del gemv, dense
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
