"""Round-trip tests for the QTEA storage format.

    python -m pytest tests            (or simply: python tests/test_pack.py)
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtea.qtea import QTEA, QTEAConfig
from qtea.pack import bit_width, pack_layer, pack_ternary, unpack_layer, unpack_ternary


def test_ternary_codes_survive_base3_packing():
    codes = torch.randint(-1, 2, (64, 512), dtype=torch.int8)
    packed = pack_ternary(codes, group_size=128)

    assert packed.shape == (64, 4 * 26)  # 128 values per group -> 26 bytes
    assert torch.equal(unpack_ternary(packed, 512, group_size=128), codes)


def test_packed_layer_reproduces_the_quantized_weight():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    layer = nn.Linear(512, 256, bias=False).to(device=device, dtype=torch.float16)
    quantizer = QTEA(layer, QTEAConfig())
    for _ in range(8):
        quantizer.add_batch(torch.randn(1, 128, 512, device=device, dtype=torch.float16))
    packed = pack_layer(quantizer.quantize())

    # `quantize` leaves the dequantized weight on the layer; unpacking the
    # checkpoint has to give exactly the same thing back.
    assert torch.equal(unpack_layer(packed).half(), layer.weight.data.cpu())
    assert set(unpack_ternary(packed["ternary"], 512, 128).unique().tolist()) <= {-1, 0, 1}
    assert bit_width(packed)["payload"] > 1.6


if __name__ == "__main__":
    test_ternary_codes_survive_base3_packing()
    test_packed_layer_reproduces_the_quantized_weight()
    print("ok")
