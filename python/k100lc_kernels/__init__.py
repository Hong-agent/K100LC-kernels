"""K100LC-kernels：gfx926 (K100_LC) 自研汇编内核与常驻 HSA 运行时。"""
from .catalog import info, kernels
from .model import (DotLinear, F32Linear, KVCache, MLP, MoECombine, RMSNorm,
                    RT4Linear, Sampler, SwiGLU, Workspace, div_magic)
from .quant import dequant_int4_group128, pack_int4_group128
from .rt4 import RT4File, RT4Tensor, W4Runner
from .runtime import Runtime

__all__ = [
    "Runtime", "kernels", "info",
    "RT4File", "RT4Tensor", "W4Runner",
    "Workspace", "DotLinear", "F32Linear", "RT4Linear", "RMSNorm", "SwiGLU",
    "MLP", "MoECombine", "KVCache", "Sampler", "div_magic",
    "pack_int4_group128", "dequant_int4_group128",
]
