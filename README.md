<div align="center">

# QTEA: Ternary LLMs with Sparse Residual Salient Weight and By-Column Optimization

<p>
  <a href="https://arxiv.org/abs/XXXX.XXXXX">
    <img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg?logo=arxiv" alt="arXiv">
  </a>
  <a href="https://github.com/Intelligent-Microsystems-Lab/QTEA">
    <img src="https://img.shields.io/github/stars/Intelligent-Microsystems-Lab/QTEA?style=social" alt="GitHub Stars">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
  </a>
</p>

<p>
  <a href="https://coco-alen.github.io/personal-web/">Yipin Guo</a>,
  Arun M George,
  Jie Fu,
  Tareq Mahmoud,
  <a href="https://siddharth-joshi.com/">Siddharth Joshi</a>
</p>

<p>University of Notre Dame</p>

</div>

---

## News

- **2026-08-21:** QTEA is accepted to the **EMNLP 2026 Main Conference**.

---

## Abstract

QTEA is a post-training quantization method that compresses the linear layers of
a decoder-only LLM to an effective **1.7 bits per weight** without any
retraining. A single pass over 256 calibration sequences is enough.

The key idea is to treat salient weights as *error compensators* rather than as
weights that bypass quantization. QTEA first quantizes every weight into a
compact ternary base `{-1, 0, +1}`, then spends a small residual budget on the
columns where ternarization hurts most — keeping one FP8 value per four rows
inside those columns, a *column-semi sparse* layout that recovers most of the
accuracy of unstructured salient weights while staying GPU-friendly. Two further
changes to the GPTQ-style column-by-column sweep make the ternary base fit
better: a per-column rescale factor refined jointly with the ternary
assignments, and an error decay that attenuates error propagation so late
columns are not over-compensated. A lookup-table CUDA kernel then evaluates the
result without ever decoding the ternary weights.

<p align="center">
  <img width="100%" src="figs/overview.png" alt="QTEA overview">
</p>

## Highlights

**Quality.** QTEA has the best average zero-shot accuracy of any sub-2-bit
method on every model evaluated, and the advantage grows with model size.

- **Qwen3-14B:** average zero-shot accuracy rises from 45.11% for the strongest
  sub-2-bit baseline to **52.65%**, a 16.7% relative gain, with **1.40×** lower
  WikiText-2 perplexity (16.48 → 11.78) and **2.61×** lower C4 perplexity
  (68.13 → 26.14).
- **Llama3-8B:** accuracy improves from 37.79% to **40.29%** (6.6% relative),
  with **1.34×** and **1.95×** lower WikiText-2 and C4 perplexity.
- The 1:4 semi-sparse residual costs only 0.9 accuracy points against an
  unstructured salient-weight upper bound, at **4× lower residual storage**.

**Efficiency.** The lookup-table kernel turns the compact representation into a
real speedup.

- **7.2×** faster per-token generation than FP16 with CUDA Graphs on Llama2-70B
  (41.13 → 5.70 ms/token), and **13.3×** against FP16 without CUDA Graphs;
  **3.62×** on Qwen3-14B.
- The sparse residual path is essentially free: latency matches a ternary-only
  kernel to within 0.01–0.03 ms/token.
- Speedup is stable at **3.07–3.20×** for batch size 1 across 512–4096 tokens of
  context, and still 2.54× at batch size 4.
- On a TSMC 22nm standard-cell implementation, **3.83×** lower latency and
  **69.4%** lower energy than dense FP16 matrix multiplication.

<p align="center">
  <img width="50%" src="figs/qwen_wikitext2_ppl_vs_model_size.png" alt="WikiText-2 perplexity vs. model size on Qwen3">
</p>

---

## Installation

```bash
conda create -n qtea python=3.12 -y
conda activate qtea
pip install -r requirements.txt
```

If you already maintain a PyTorch CUDA environment, install the packages from
`requirements.txt` there instead of creating a new one. `lm_eval` is only needed
for zero-shot accuracy, and a CUDA toolchain (`nvcc`) is only needed for the
lookup-table inference kernel.

## Quantization

To quantize a model yourself:

```bash
python quantize.py --model Qwen/Qwen3-8B-Base --output packed/qwen3-8b.pt
```

This writes a packed checkpoint holding the ternary codes, the residuals and the
scales — nothing is stored in full precision. Only one decoder block is resident
on the GPU at a time, so the footprint is set by the calibration activations and
the largest block rather than by the model. Lower `--nsamples` if OOM;

A QTEA-quantized **Qwen3-14B-Base** checkpoint is also published on the Hugging Face
Hub as [`ims-lab/Qwen3-14B-base-QTEA`](https://huggingface.co/ims-lab/Qwen3-14B-base-QTEA),
so you can skip this step and go straight to [Evaluation](#evaluation):

```bash
huggingface-cli download ims-lab/Qwen3-14B-base-QTEA packed_qwen3_14b.pt --local-dir packed/
```

## Evaluation

```bash
python eval/evaluate.py --model Qwen/Qwen3-8B-Base --checkpoint packed/qwen3-8b.pt
```

Reports WikiText-2 and C4 perplexity plus zero-shot accuracy on the seven tasks
above. Drop `--checkpoint` for the FP16 baseline, and add `--backend lut` to run
the packed weights through the lookup-table CUDA kernel instead of rebuilding
dense FP16 weights. See [`eval/README.md`](eval/README.md) for the full set of
options and for how the kernel works.

## Reproducing the Paper

```bash
bash scripts/quantize_qwen3.sh          # Qwen3-Base, 0.6B to 14B
bash scripts/quantize_llama3.sh         # Llama3-8B
```

Both scripts quantize and then evaluate, writing JSON results to `results/`. The
defaults in `quantize.py` are the recipe used for every number in the paper;
Llama3 is the one exception and takes an extra ternary-fitting round, which the
script passes for you.

Reproduced here on one H200 with torch 2.8 / transformers 4.56:

| model | WikiText-2 | C4 | zero-shot avg |
| --- | --- | --- | --- |
| Qwen3-0.6B-Base | 63.58 | 172.66 | 34.37 |
| Qwen3-14B-Base | 11.78 | 26.14 | 52.67 |
| Llama3-8B | 24.09 | 66.39 | 40.13 |

Quantization is deterministic for a given GPU and library stack, but it is not
robust to last-bit arithmetic differences between environments: the ternary
assignment is a hard threshold inside a sequential error-feedback loop, so a
handful of flipped codes early on can move perplexity by a percent or more.

Llama-3 and Qwen3 are supported out of the box. Any decoder stack that exposes
`model.model.layers` with the usual seven projections works unchanged; other
families need their projection names added to `qtea/sequential.py`.

---

## Repository Contents

**Quantization**

- `quantize.py`: command-line entry point — quantizes one model and saves the
  packed checkpoint.
- `qtea/qtea.py`: the QTEA algorithm for a single linear layer. Sweeps the
  columns left to right, picking the salient columns, refining their rescale
  factors, fitting the sparse residual and propagating the quantization error.
- `qtea/quantizer.py`: fits the ternary scale and centre that every group of 128
  columns shares.
- `qtea/sequential.py`: applies the quantizer to the whole decoder stack, one
  block at a time, so only one block ever sits on the GPU.
- `qtea/pack.py`: the checkpoint format — packs and unpacks the ternary codes
  (five per byte), the FP8 residuals and the scales.
- `qtea/data.py`: WikiText-2 and C4 loaders for calibration and perplexity.
- `qtea/model.py`: HuggingFace model loading.

**Evaluation**

- `eval/evaluate.py`: measures perplexity and zero-shot accuracy for a packed
  checkpoint, or for the FP16 baseline.
- `eval/perplexity.py`: perplexity, computed one decoder block at a time to keep
  memory low.
- `eval/packed_linear.py`: loads a packed checkpoint into a HuggingFace model.
- `eval/ternary_kernel.py` and `eval/csrc/ternary_gemv.cu`: the lookup-table
  GEMV kernel — the Python wrapper and the CUDA source behind the paper's
  latency numbers.
- `eval/check_kernel.py`: checks the kernel against a dense FP16 matmul.

**Other**

- `scripts/`: one reproduction script per model family.
- `tests/`: round-trip tests for the checkpoint format (`python -m pytest tests`).
- `figs/`: figures used by this README and by `eval/README.md`.

## Citation

If you find QTEA useful in your research, please cite:

```bibtex

```

## License

This project is released under the MIT License. See `LICENSE` for details. The
evaluation datasets and the models retain their own licences.
