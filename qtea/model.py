"""Model loading shared by quantization and evaluation."""

import torch
from transformers import AutoModelForCausalLM

DTYPES = {"auto": "auto", "float16": torch.float16, "bfloat16": torch.bfloat16}


def load_model(name, dtype="float16", seqlen=2048):
    """Load a causal LM on CPU, together with the sequence length to evaluate at.

    `auto` keeps the dtype the checkpoint was published in; the paper quantizes
    Qwen3 in FP16 and Llama3 in its native BF16.
    """
    model = AutoModelForCausalLM.from_pretrained(name, dtype=DTYPES[dtype], low_cpu_mem_usage=True)
    model.eval()
    return model, min(model.config.max_position_embeddings, seqlen)
