"""Perplexity over a held-out token stream, one decoder block at a time.

Streaming keeps at most one block on the GPU, so a 14B model can be evaluated
without holding the whole network in device memory.
"""

import torch
import torch.nn as nn
from tqdm import tqdm


@torch.no_grad()
def perplexity(model, tokens, device, seqlen):
    from transformers.cache_utils import DynamicCache

    windows = tokens.numel() // seqlen
    tokens = tokens[:, : windows * seqlen].reshape(windows, seqlen)
    use_cache, model.config.use_cache = model.config.use_cache, False

    blocks = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    model.model.rotary_emb = model.model.rotary_emb.to(device)

    hidden = torch.stack([
        model.model.embed_tokens(window.to(device)) for window in tokens
    ])
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    position_ids = torch.arange(seqlen, device=device).unsqueeze(0)
    for block in tqdm(blocks, desc="blocks", leave=False):
        block = block.to(device)
        for i in range(windows):
            sample = hidden[i : i + 1]
            hidden[i] = block(
                sample,
                attention_mask=None,
                position_ids=position_ids,
                position_embeddings=model.model.rotary_emb(sample, position_ids),
                past_key_values=DynamicCache(),
            )[0]
        block.cpu()
        torch.cuda.empty_cache()

    model.model.norm = model.model.norm.to(device)
    model.lm_head = model.lm_head.to(device)

    loss_fn, total = nn.CrossEntropyLoss(), 0.0
    for i in range(windows):
        logits = model.lm_head(model.model.norm(hidden[i : i + 1]))
        loss = loss_fn(logits[:, :-1].flatten(0, 1).float(), tokens[i, 1:].to(device))
        total += loss.item() * seqlen

    model.model.norm = model.model.norm.cpu()
    model.lm_head = model.lm_head.cpu()
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    return float(torch.exp(torch.tensor(total / (windows * seqlen))))
