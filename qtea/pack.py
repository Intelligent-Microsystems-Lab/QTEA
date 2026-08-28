"""Storage format for a QTEA-quantized layer.

Ternary codes are stored in base 3, five per byte (3^5 = 243 < 256), which costs
1.6 bits per weight.  The residual of a salient column keeps one FP8 (E4M3)
value per group of four rows together with a 2-bit index, i.e. 2.5 bits per
weight of that column; `residual_scale` is the multiplier that maps the column
onto the FP8 range.  Everything else is per-group or per-column metadata.
"""

import os

import torch

_POWERS = torch.tensor([1, 3, 9, 27, 81], dtype=torch.int32)


def pack_ternary(codes, group_size):
    """Pack int8 ternary codes into base-3 bytes, five per byte.

    Packing restarts at every group boundary so that a byte never spans two
    quantization groups; a group of 128 columns therefore uses 26 bytes.
    """
    rows, cols = codes.shape
    bytes_per_group = (group_size + 4) // 5
    digits = (codes.int() + 1)  # {-1, 0, +1} -> {0, 1, 2}

    groups = digits.reshape(rows, cols // group_size, group_size)
    padding = bytes_per_group * 5 - group_size
    if padding:
        groups = torch.cat([groups, torch.ones(rows, groups.shape[1], padding, dtype=torch.int32)], dim=2)
    groups = groups.reshape(rows, -1, 5)
    return (groups * _POWERS).sum(dim=2).to(torch.uint8).reshape(rows, -1)


def unpack_ternary(packed, cols, group_size):
    """Inverse of `pack_ternary`."""
    rows = packed.shape[0]
    bytes_per_group = (group_size + 4) // 5

    values = packed.reshape(rows, -1, 1).int()
    digits = (values // _POWERS) % 3 - 1
    digits = digits.reshape(rows, -1, bytes_per_group * 5)[:, :, :group_size]
    return digits.reshape(rows, cols).to(torch.int8)


def pack_layer(payload):
    """Turn the output of `QTEA.quantize` into its on-disk representation."""
    rows, group_size = payload["rows"], payload["group_size"]
    index = payload["residual_index"]  # (n_salient, rows // residual_group), values 0..3
    scale = payload["residual_scale"]

    if index.numel():
        pad = -index.shape[1] % 4
        if pad:
            index = torch.cat([index, torch.zeros(index.shape[0], pad, dtype=index.dtype)], dim=1)
        pairs = index.reshape(index.shape[0], -1, 4).to(torch.uint8)
        index_packed = pairs[:, :, 0] | (pairs[:, :, 1] << 2) | (pairs[:, :, 2] << 4) | (pairs[:, :, 3] << 6)
        value_fp8 = (payload["residual_value"] * scale[:, None]).to(torch.float8_e4m3fn)
    else:
        index_packed = torch.zeros(0, 0, dtype=torch.uint8)
        value_fp8 = torch.zeros(0, 0, dtype=torch.float8_e4m3fn)

    return {
        "rows": rows,
        "cols": payload["cols"],
        "group_size": group_size,
        "residual_group": payload["residual_group"],
        "ternary": pack_ternary(payload["ternary"], group_size),
        "col_v": payload["col_v"].float(),
        "group_alpha": payload["group_alpha"].float(),
        "group_beta": payload["group_beta"].float(),
        "salient_cols": payload["salient_cols"].to(torch.int32),
        "residual_index": index_packed,
        "residual_value": value_fp8,
        "residual_scale": scale.float(),
    }


def residual_columns(layer):
    """Dense (n_salient, rows) residual of a packed layer, in FP32."""
    if layer["residual_value"].numel() == 0:
        return torch.zeros(0, layer["rows"])

    packed = layer["residual_index"].int()
    index = torch.stack([(packed >> shift) & 0x3 for shift in (0, 2, 4, 6)], dim=2).reshape(packed.shape[0], -1)
    value = layer["residual_value"].float() / layer["residual_scale"][:, None]

    dense = torch.zeros(value.shape[0], value.shape[1], layer["residual_group"])
    dense.scatter_(2, index[:, : value.shape[1], None].long(), value[:, :, None])
    return dense.reshape(value.shape[0], -1)


def unpack_layer(layer):
    """Reconstruct the dense FP32 weight of a packed layer."""
    codes = unpack_ternary(layer["ternary"], layer["cols"], layer["group_size"]).float()
    alpha = layer["group_alpha"].repeat_interleave(layer["group_size"], dim=1)
    beta = layer["group_beta"].repeat_interleave(layer["group_size"], dim=1)

    weight = alpha * layer["col_v"][None, :] * codes + beta
    salient = layer["salient_cols"].long()
    if salient.numel():
        weight[:, salient] += residual_columns(layer).t()
    return weight


def bit_width(layer):
    """Bits per weight, split into payload and per-group metadata."""
    rows, cols = layer["rows"], layer["cols"]
    n_salient, n_groups = layer["salient_cols"].numel(), layer["group_alpha"].shape[1]

    ternary = layer["ternary"].numel() * 8
    residual = n_salient * (rows // layer["residual_group"]) * 10  # 8-bit value + 2-bit index
    rescale = cols * 32                                            # col_v, FP32
    metadata = 2 * rows * n_groups * 32                            # alpha and beta, FP32
    return {
        "payload": (ternary + residual + rescale) / (rows * cols),
        "metadata": metadata / (rows * cols),
    }


def save(layers, model_name, path):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    torch.save({"model_name": model_name, "layers": layers}, path)


def load(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoint["layers"], checkpoint["model_name"]
