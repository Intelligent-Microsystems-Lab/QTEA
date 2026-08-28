"""Lookup-table CUDA GEMV for QTEA-packed layers.

The kernel builds, for every quantization group, a table of the dot products
between all packed ternary patterns and the rescaled activations v_j * x_j, then
turns the ternary matrix-vector product into table lookups.  The sparse FP8
residual of the salient columns is accumulated in the same kernel.

The extension is compiled on first use with `torch.utils.cpp_extension.load`,
which needs `nvcc` on PATH.
"""

import os

import torch
from torch.utils.cpp_extension import load

_extension = None


def kernel():
    global _extension
    if _extension is None:
        _extension = load(
            name="qtea_ternary_gemv",
            sources=[os.path.join(os.path.dirname(__file__), "csrc", "ternary_gemv.cu")],
            extra_cuda_cflags=["--use_fast_math"],
            verbose=False,
        )
    return _extension


class TernaryGEMV(torch.nn.Module):
    """`y = x @ W^T` for one packed layer, without ever materialising W."""

    def __init__(self, layer):
        super().__init__()
        from qtea.pack import residual_columns

        rows, cols = int(layer["rows"]), int(layer["cols"])
        self.rows, self.cols = rows, cols

        packed = layer["ternary"].contiguous()
        alpha, beta = layer["group_alpha"].float(), layer["group_beta"].float()
        n_groups = alpha.shape[1]
        bytes_per_group = packed.shape[1] // n_groups

        self.register_buffer("packed", packed)
        self.register_buffer("packed_t", packed.reshape(rows, n_groups, bytes_per_group)
                             .permute(1, 2, 0).contiguous())
        self.register_buffer("col_v", layer["col_v"].float().contiguous())
        self.register_buffer("alpha", alpha.contiguous())
        self.register_buffer("beta", beta.contiguous())

        # The batch-1 kernels sweep rows for a fixed (group, byte); transposed
        # copies make those reads coalesced.  They only pay off for wide layers.
        transposed = n_groups >= 32 or rows > 1024
        self.register_buffer("alpha_t", alpha.t().half().contiguous() if transposed
                             else torch.zeros(0, 0, dtype=torch.float16))
        self.register_buffer("beta_t", beta.t().half().contiguous() if transposed
                             else torch.zeros(0, 0, dtype=torch.float16))

        salient = layer["salient_cols"].to(torch.int32)
        values = residual_columns(layer).float().contiguous()  # (n_salient, rows)
        self.register_buffer("salient_cols", salient)
        self.register_buffer("salient_values", values)
        self.register_buffer("salient_values_t", values.t().half().contiguous()
                             if salient.numel() >= 1024 else torch.zeros(rows, 0, dtype=torch.float16))

        # Scratch space, allocated once so that the forward pass is capture-safe.
        self.register_buffer("lut", torch.empty(n_groups, bytes_per_group, 122, dtype=torch.float16))
        self.register_buffer("activation_sums", torch.empty(n_groups, dtype=torch.float32))
        self.register_buffer("out_f32", torch.zeros(rows, dtype=torch.float32))
        self.register_buffer("out_f16", torch.empty(rows, dtype=torch.float16))

    def forward(self, x):
        y = kernel().ternary_gemv_fused(
            self.packed, self.packed_t, self.col_v,
            self.alpha, self.beta, self.alpha_t, self.beta_t,
            x, self.salient_cols, self.salient_values, self.salient_values_t,
            self.lut, self.activation_sums, self.out_f32, self.out_f16,
        )
        return y.reshape(*x.shape[:-1], self.rows)
