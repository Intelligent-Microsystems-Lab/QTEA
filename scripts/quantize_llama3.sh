#!/usr/bin/env bash
# Reproduce the Llama3-8B row of Table 1.
#
# Llama3 keeps its native bfloat16 activations and takes one extra iterative
# ternary fitting round; everything else is the default recipe.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p packed results

model="${1:-meta-llama/Meta-Llama-3-8B}"
checkpoint="packed/llama3-8b.pt"

python quantize.py --model "$model" --output "$checkpoint" \
    --dtype bfloat16 --itf-iters 2
python eval/evaluate.py --model "$model" --checkpoint "$checkpoint" \
    --dtype bfloat16 --output "results/llama3-8b.json"
