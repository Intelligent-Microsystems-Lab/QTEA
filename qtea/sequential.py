"""Layer-by-layer quantization driver for Llama-3 and Qwen3 decoder stacks.

Only one decoder block is on the GPU at a time.  Each block is first run on the
calibration activations to accumulate a Hessian per projection, then quantized,
then re-run so that the next block sees the *quantized* activations.
"""

import gc

import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache

from .gptq import QTEA, QTEAConfig
from .pack import bit_width, pack_layer

PROJECTIONS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def linear_layers(block):
    """The projections of one decoder block, keyed by their dotted name."""
    found = {
        name: module
        for name, module in block.named_modules()
        if isinstance(module, nn.Linear) and name.split(".")[-1] in PROJECTIONS
    }
    if not found:
        raise ValueError(
            f"no known projection in {type(block).__name__}; add its names to PROJECTIONS"
        )
    return found


@torch.no_grad()
def capture_block_inputs(model, dataloader, device):
    """Run the embedding stack to collect the inputs of decoder block 0.

    A pre-hook records what block 0 is called with and aborts the forward pass,
    so the keyword arguments (attention mask, cache position, ...) that the
    model builds for us can be reused when blocks are driven directly.
    """
    blocks = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    blocks[0] = blocks[0].to(device)

    store = {"inputs": [], "kwargs": None}

    def capture(_module, args, kwargs):
        store["inputs"].append(args[0] if args else kwargs["hidden_states"])
        store["kwargs"] = kwargs
        raise StopIteration

    handle = blocks[0].register_forward_pre_hook(capture, with_kwargs=True)
    for batch in dataloader:
        try:
            model(batch.to(device))
        except StopIteration:
            pass
    handle.remove()

    blocks[0] = blocks[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()
    return torch.cat(store["inputs"]), store["kwargs"]


def block_forward(model, block, hidden_states, kwargs, device):
    """Run one decoder block on a single calibration sample."""
    position_ids = torch.arange(hidden_states.shape[1], device=device).unsqueeze(0)
    position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
    return block(
        hidden_states,
        attention_mask=kwargs.get("attention_mask"),
        position_ids=position_ids,
        position_embeddings=position_embeddings,
        past_key_values=DynamicCache(),
    )[0]


@torch.no_grad()
def quantize_model(model, dataloader, device, config: QTEAConfig):
    """Quantize every decoder projection and return the packed layers."""
    if not hasattr(getattr(model, "model", None), "layers"):
        raise ValueError(f"{type(model).__name__} has no model.model.layers decoder stack")

    use_cache, model.config.use_cache = model.config.use_cache, False
    blocks = model.model.layers

    model.model.rotary_emb = model.model.rotary_emb.to(device)
    inputs, kwargs = capture_block_inputs(model, dataloader, device)
    outputs = torch.empty_like(inputs)

    packed, payload_bits, metadata_bits, total_weights = {}, 0.0, 0.0, 0
    for index in range(len(blocks)):
        block = blocks[index].to(device)
        targets = linear_layers(block)
        quantizers = {name: QTEA(module, config) for name, module in targets.items()}

        handles = [
            module.register_forward_pre_hook(
                lambda _module, args, name=name: quantizers[name].add_batch(args[0])
            )
            for name, module in targets.items()
        ]
        for i in range(inputs.shape[0]):
            block_forward(model, block, inputs[i : i + 1], kwargs, device)
        for handle in handles:
            handle.remove()

        for name, quantizer in quantizers.items():
            print(f"[block {index:>3}] {name}")
            layer = pack_layer(quantizer.quantize())
            quantizer.free()
            packed[f"model.layers.{index}.{name}"] = layer

            bits, weights = bit_width(layer), layer["rows"] * layer["cols"]
            payload_bits += bits["payload"] * weights
            metadata_bits += bits["metadata"] * weights
            total_weights += weights

        for i in range(inputs.shape[0]):
            outputs[i] = block_forward(model, block, inputs[i : i + 1], kwargs, device)

        blocks[index] = block.cpu()
        del block, quantizers
        gc.collect()
        torch.cuda.empty_cache()
        inputs, outputs = outputs, inputs

    model.model.rotary_emb = model.model.rotary_emb.cpu()
    model.config.use_cache = use_cache
    print(
        f"\nquantized {total_weights / 1e9:.2f}B weights at "
        f"{payload_bits / total_weights:.3f} bpw payload "
        f"+ {metadata_bits / total_weights:.3f} bpw group scales"
    )
    return packed
