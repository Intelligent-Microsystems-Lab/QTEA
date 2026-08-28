"""QTEA: ternary post-training quantization with sparse residual compensation."""

from .qtea import QTEA, QTEAConfig
from .sequential import quantize_model

__all__ = ["QTEA", "QTEAConfig", "quantize_model"]
