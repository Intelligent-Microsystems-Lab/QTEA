"""Install a QTEA checkpoint into a HuggingFace model.

Two backends are available:

  ``dequant``  reconstructs dense FP16 weights.  Numerically identical to what
               the quantizer produced, and the fastest way to reproduce the
               accuracy numbers.
  ``lut``      keeps the weights packed and runs the custom CUDA kernel.  Uses
               far less memory and is the path the latency numbers come from.
"""

import gc

import torch
import torch.nn as nn

from qtea.pack import load, unpack_layer


class PackedTernaryLinear(nn.Module):
    """Drop-in `nn.Linear` backed by the packed ternary CUDA kernel."""

    def __init__(self, layer, bias=None):
        super().__init__()
        from ternary_kernel import TernaryGEMV

        self.in_features, self.out_features = int(layer["cols"]), int(layer["rows"])
        self.gemv = TernaryGEMV(layer)
        self.register_buffer("bias", None if bias is None else bias.detach().clone())

    def forward(self, x):
        y = self.gemv(x)
        return y if self.bias is None else y + self.bias.to(y.dtype)


def _parent_of(model, dotted_name):
    """The module holding `dotted_name`, and the attribute name inside it."""
    *path, leaf = dotted_name.split(".")
    parent = model
    for part in path:
        parent = getattr(parent, part)
    return parent, leaf


def install_checkpoint(model, path, backend="dequant"):
    """Replace every quantized projection of `model` with its packed version."""
    layers, model_name = load(path)
    print(f"installing {len(layers)} quantized projections from {path} ({backend} backend)")

    for name, layer in layers.items():
        parent, leaf = _parent_of(model, name)
        linear = getattr(parent, leaf)

        if backend == "dequant":
            weight = unpack_layer(layer).reshape(linear.weight.shape)
            linear.weight.data = weight.to(linear.weight.dtype)
        else:
            bias = None if linear.bias is None else linear.bias.detach()
            setattr(parent, leaf, PackedTernaryLinear(layer, bias))

    gc.collect()
    torch.cuda.empty_cache()
    return model, model_name
