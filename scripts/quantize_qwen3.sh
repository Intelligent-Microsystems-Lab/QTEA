#!/usr/bin/env bash
# Reproduce the Qwen3-Base rows of Table 1.
#
#   bash scripts/quantize_qwen3.sh 8B          # one size
#   bash scripts/quantize_qwen3.sh             # every size in the table
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p packed results

sizes=("$@")
if [ ${#sizes[@]} -eq 0 ]; then
    sizes=(0.6B 1.7B 4B 8B 14B)
fi

for size in "${sizes[@]}"; do
    model="Qwen/Qwen3-${size}-Base"
    tag="qwen3-$(echo "$size" | tr '[:upper:]' '[:lower:]')"

    python quantize.py --model "$model" --output "packed/${tag}.pt"
    python eval/evaluate.py --model "$model" --checkpoint "packed/${tag}.pt" \
        --output "results/${tag}.json"
done
