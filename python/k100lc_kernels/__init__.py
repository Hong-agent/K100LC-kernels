"""K100LC-kernels：gfx926 (K100_LC) 自研汇编内核与常驻 HSA 运行时。"""
from .catalog import info, kernels
from .model import (Attention, DotLinear, F32Linear, FlashAttention,
                    Int4Linear, KVCache, MLP, TransformerLayer,
                    MoECombine, MoEExperts, RMSNorm, RT4Linear, RoPE, Sampler,
                    SwiGLU, Workspace, div_magic, run_sequence)
from .quant import (ct_int4_to_rt4, ct_int4_to_twos_complement,
                    dequant_int4_group128, int4_scale_group_first_f32,
                    pack_int4_group128)
from .rt4 import RT4File, RT4Tensor, W4Runner
from .runtime import Runtime

__all__ = [
    "Runtime", "kernels", "info",
    "RT4File", "RT4Tensor", "W4Runner",
    "Workspace", "DotLinear", "F32Linear", "RT4Linear", "Int4Linear",
    "Attention", "FlashAttention", "TransformerLayer",
    "RMSNorm", "SwiGLU", "MLP", "MoECombine", "MoEExperts", "RoPE", "KVCache",
    "Sampler", "div_magic", "run_sequence",
    "pack_int4_group128", "dequant_int4_group128",
    "ct_int4_to_rt4", "ct_int4_to_twos_complement", "int4_scale_group_first_f32",
]
