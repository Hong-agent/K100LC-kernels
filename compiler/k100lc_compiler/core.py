#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K100LC 内核编译器：受限 Python DSL → gfx926 汇编 → HSACO。

支持子集（v1）：
  * 内核/参数：`name: ptr[f32]`、`name: u32/s32/f32`
  * 内建：gid()/tid()/bid()/lane()、exp/sqrt/rsqrt/fma/fabs/max
  * 语句：赋值、增强赋值、if（无 else 或 uniform else）、for range、
    break/continue、指针 load/store
  * 类型：f32/u32/s32；指针元素类型 f32/u32/s32

示例见 compiler/examples/。生成的内核使用标准隐藏 kernarg
(hidden_block_count_x / hidden_group_size_x ...)，因此 gid() 可用于跨 workgroup。
"""
from __future__ import annotations

import ast
import dataclasses
import pathlib
import struct
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import asm  # noqa: E402
from make_hsaco_multi import build_elf  # noqa: E402


class CompileError(Exception):
    pass


def _align(n: int, a: int) -> int:
    return (n + a - 1) // a * a


def _ty_name(node) -> str:
    """解析注解：'ptr[f32]' / Name('u32') / Str。"""
    if isinstance(node, ast.Str):
        return node.s
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Subscript):
        base = _ty_name(node.value)
        idx = _ty_name(node.slice)
        return f"{base}[{idx}]"
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    raise CompileError(f"不支持的注解 {ast.dump(node)}")


def parse_type(text: str) -> tuple[str, bool]:
    """返回 (类型, 是否指针)。"""
    if text.startswith("ptr[") and text.endswith("]"):
        return text[4:-1], True
    return text, False


@dataclasses.dataclass
class Param:
    name: str
    ty: str
    is_ptr: bool
    off: int


@dataclasses.dataclass
class Val:
    kind: str          # v/s/lit/pred
    reg: int | None
    ty: str
    value: int | float | None = None


class CodeGen:
    def __init__(self, fn: ast.FunctionDef):
        self.fn = fn
        self.lines: list[str] = []
        self.params: list[Param] = []
        self.env: dict[str, Val] = {}
        self.var_v = 2
        self.var_s = 16
        self.tmp_v = 64
        self.tmp_s = 64
        self.addr_v = 200
        self.save_s = 48
        self.max_v = 1
        self.max_s = 15
        self.loop_stack: list[tuple[str, str]] = []
        self.zero_v = 1
        self.label_n = 0
        self._parse_params()

    # ---------------- 基础 ----------------
    def emit(self, text: str) -> None:
        self.lines.append(text)

    def label(self, name: str) -> None:
        self.lines.append(f"{name}:")

    def new_label(self, prefix: str) -> str:
        self.label_n += 1
        return f"{prefix}_{self.label_n}"

    def alloc_tmp_v(self) -> int:
        r = self.tmp_v
        self.tmp_v += 1
        if self.tmp_v > 255:
            raise CompileError("VGPR 溢出（v1 不做 spill）")
        self.max_v = max(self.max_v, r)
        return r

    def free_tmp_v(self, n: int = 1) -> None:
        # v1：不做临时寄存器复用，避免结果寄存器被后续表达式覆盖。
        return

    def alloc_addr_pair(self) -> tuple[int, int]:
        lo = self.addr_v
        self.addr_v += 2
        if self.addr_v > 255:
            raise CompileError("地址 VGPR 溢出")
        self.max_v = max(self.max_v, lo + 1)
        return lo, lo + 1

    def free_addr_pair(self) -> None:
        return

    def _parse_params(self) -> None:
        off = 0
        for a in self.fn.args.args:
            if a.annotation is None:
                raise CompileError(f"{self.fn.name}: 参数 {a.arg} 缺类型注解")
            ty, is_ptr = parse_type(_ty_name(a.annotation))
            if is_ptr:
                off = _align(off, 8)
                p = Param(a.arg, ty, True, off)
                off += 8
            else:
                p = Param(a.arg, ty, False, off)
                off += 4
            self.params.append(p)
        self.explicit_end = off
        self.hidden_off = _align(off, 8)

    def compile(self) -> str:
        self.emit(".text")
        self.emit(f"k_{self.fn.name}:")
        self.emit(f"v_mov_b32_e32 v{self.zero_v}, 0")
        for p in self.params:
            if p.is_ptr:
                s = self._alloc_var_s(2)
                self.emit(f"s_load_dwordx2 s[{s}:{s + 1}], s[4:5], 0x{p.off:x}")
                self.env[p.name] = Val("ptr", s, f"ptr:{p.ty}")
                self.max_s = max(self.max_s, s + 1)
            elif p.ty == "f32":
                # f32 标量：先放 SGPR，使用时再搬到 VGPR
                s = self._alloc_var_s()
                self.emit(f"s_load_dword s{s}, s[4:5], 0x{p.off:x}")
                self.env[p.name] = Val("s", s, "f32")
                self.max_s = max(self.max_s, s)
            else:
                s = self._alloc_var_s()
                self.emit(f"s_load_dword s{s}, s[4:5], 0x{p.off:x}")
                self.env[p.name] = Val("s", s, p.ty)
                self.max_s = max(self.max_s, s)
        self.emit("s_waitcnt lgkmcnt(0)")
        for st in self.fn.body:
            self.stmt(st)
        self.emit("s_endpgm")
        return "\n".join(self.lines) + "\n"

    def _alloc_var_v(self) -> int:
        r = self.var_v
        self.var_v += 1
        if self.var_v > 127:
            raise CompileError("变量 VGPR 超过 127")
        self.max_v = max(self.max_v, r)
        return r

    def _alloc_var_s(self, size: int = 1) -> int:
        r = self.var_s
        if size == 2 and r % 2:
            r += 1
        self.var_s = r + size
        if self.var_s > 47:
            raise CompileError("变量 SGPR 超过 47")
        self.max_s = max(self.max_s, r + size - 1)
        return r

    def _alloc_tmp_s(self) -> int:
        r = self.tmp_s
        self.tmp_s += 1
        if self.tmp_s > 101:
            raise CompileError("临时 SGPR 超过 101")
        self.max_s = max(self.max_s, r)
        return r

    # ---------------- 值与类型 ----------------
    def is_uniform(self, v: Val) -> bool:
        return v.kind in ("s", "lit") and v.ty != "f32"

    def as_vreg(self, v: Val) -> int:
        if v.kind == "v":
            return v.reg
        r = self.alloc_tmp_v()
        if v.kind == "s":
            self.emit(f"v_mov_b32_e32 v{r}, s{v.reg}")
        elif v.kind == "lit":
            if isinstance(v.value, float):
                bits = struct.unpack("<I", struct.pack("<f", v.value))[0]
                self.emit(f"v_mov_b32_e32 v{r}, 0x{bits:08x}")
            else:
                self.emit(f"v_mov_b32_e32 v{r}, {int(v.value)}")
        else:
            raise CompileError(f"不能转 VGPR: {v}")
        return r

    def as_sreg(self, v: Val) -> int:
        if v.kind == "s":
            return v.reg
        if v.kind == "lit":
            r = self._alloc_tmp_s()
            self.emit(f"s_mov_b32 s{r}, {int(v.value)}")
            return r
        raise CompileError(f"不是 uniform 值: {v}")

    def type_of(self, node) -> str:
        if isinstance(node, ast.Name):
            if node.id in self.env:
                return self.env[node.id].ty
            return "u32"
        if isinstance(node, ast.Constant):
            if isinstance(node.value, float):
                return "f32"
            return "u32"
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                n = node.func.id
                if n in ("exp", "sqrt", "rsqrt", "fma", "fabs", "max", "min"):
                    return "f32"
                if n in ("gid", "tid", "bid", "lane"):
                    return "u32"
                if n in ("f32", "u32", "s32"):
                    return n
        return "u32"

    # ---------------- 表达式 ----------------
    def expr(self, node) -> Val:
        if isinstance(node, ast.Constant):
            return Val("lit", None, self.type_of(node), node.value)
        if isinstance(node, ast.Name):
            if node.id in self.env:
                return self.env[node.id]
            raise CompileError(f"未定义变量 {node.id}")
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                v = self.expr(node.operand)
                if v.kind == "lit":
                    return Val("lit", None, v.ty, -v.value)
                if v.ty != "f32":
                    z = Val("lit", None, "u32", 0)
                    return self.binop("-", z, v)
                z = Val("lit", None, "f32", 0.0)
                return self.binop("-", z, v)
            raise CompileError("只支持一元负号")
        if isinstance(node, ast.BinOp):
            a, b = self.expr(node.left), self.expr(node.right)
            op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
                  ast.LShift: "<<", ast.BitAnd: "&", ast.BitOr: "|"}.get(type(node.op))
            if op is None:
                raise CompileError(f"不支持的运算符 {type(node.op).__name__}")
            ty = "f32" if ("f32" in (a.ty, b.ty) and op in ("+", "-", "*", "/")) else \
                (a.ty if a.ty != "u32" else b.ty)
            return self.binop(op, a, b, ty)
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1:
                raise CompileError("只支持单比较")
            a, b = self.expr(node.left), self.expr(node.comparators[0])
            op = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
                  ast.Eq: "==", ast.NotEq: "!="}[type(node.ops[0])]
            return self.compare(op, a, b)
        if isinstance(node, ast.Subscript):
            return self.load(node)
        if isinstance(node, ast.Call):
            return self.call(node)
        raise CompileError(f"不支持的表达式 {ast.dump(node)}")

    def binop(self, op: str, a: Val, b: Val, ty: str | None = None) -> Val:
        ty = ty or (a.ty if a.ty != "u32" else b.ty)
        if ty == "f32":
            av, bv = self.as_vreg(a), self.as_vreg(b)
            dst = self.alloc_tmp_v()
            if op == "+":
                self.emit(f"v_add_f32_e32 v{dst}, v{av}, v{bv}")
            elif op == "-":
                self.emit(f"v_sub_f32_e32 v{dst}, v{av}, v{bv}")
            elif op == "*":
                self.emit(f"v_mul_f32_e32 v{dst}, v{av}, v{bv}")
            elif op == "/":
                if a.kind == "lit" and b.kind == "lit":
                    if float(b.value) == 0.0:
                        raise CompileError("除零常量")
                    return Val("lit", None, "f32", float(a.value) / float(b.value))
                if b.kind == "lit":
                    inv = Val("lit", None, "f32", 1.0 / float(b.value))
                    iv = self.as_vreg(inv)
                    self.emit(f"v_mul_f32_e32 v{dst}, v{av}, v{iv}")
                else:
                    t = self.alloc_tmp_v()
                    self.emit("s_nop 0")
                    self.emit(f"v_rcp_f32_e32 v{t}, v{bv}")
                    self.emit("s_nop 0")       # v_rcp_f32 后接 VALU 读的 hazard
                    self.emit(f"v_mul_f32_e32 v{dst}, v{av}, v{t}")
                    self.free_tmp_v()
            else:
                raise CompileError(f"f32 不支持 {op}")
            return Val("v", dst, "f32")
        if self.is_uniform(a) and self.is_uniform(b):
            sa, sb = self.as_sreg(a), self.as_sreg(b)
            dst = self._alloc_tmp_s()
            m = {"+": "s_add_u32", "-": "s_sub_u32", "&": "s_and_b32",
                 "|": "s_or_b32", "^": "s_xor_b32", "*": "s_mul_i32"}
            if op not in m:
                raise CompileError(f"uniform 不支持 {op}")
            self.emit(f"{m[op]} s{dst}, s{sa}, s{sb}")
            return Val("s", dst, ty)
        av, bv = self.as_vreg(a), self.as_vreg(b)
        dst = self.alloc_tmp_v()
        m = {"+": "v_add_u32_e32", "-": "v_sub_u32_e32", "*": "v_mul_lo_u32",
             "&": "v_and_b32_e32", "|": "v_or_b32_e32", "^": "v_xor_b32_e32"}
        if op == "<<":
            self.emit(f"v_lshlrev_b32_e32 v{dst}, v{av}, v{bv}")
        elif op in m:
            self.emit(f"{m[op]} v{dst}, v{av}, v{bv}")
        else:
            raise CompileError(f"int 不支持 {op}")
        return Val("v", dst, ty)

    def compare(self, op: str, a: Val, b: Val) -> Val:
        ty = "f32" if "f32" in (a.ty, b.ty) else ("s32" if "s32" in (a.ty, b.ty) else "u32")
        if self.is_uniform(a) and self.is_uniform(b) and ty != "f32":
            sa, sb = self.as_sreg(a), self.as_sreg(b)
            base = "i32" if ty == "s32" else "u32"
            m = {"<": ("s_cmp_lt", False), "<=": ("s_cmp_ge", True),
                 ">": ("s_cmp_gt", False), ">=": ("s_cmp_ge", False),
                 "==": ("s_cmp_eq", False)}
            if op == "!=":
                raise CompileError("uniform != 暂不支持")
            mn, swap = m[op]
            self.emit(f"{mn}_{base} s{sb if swap else sa}, s{sa if swap else sb}")
            return Val("spred", None, "bool")
        av, bv = self.as_vreg(a), self.as_vreg(b)
        if ty == "f32":
            m = {"<": ("v_cmp_lt_f32_e32", False), "<=": ("v_cmp_gt_f32_e32", True),
                 ">": ("v_cmp_gt_f32_e32", False), ">=": ("v_cmp_ge_f32_e32", False),
                 "==": ("v_cmp_eq_f32_e32", False), "!=": ("v_cmp_neq_f32_e32", False)}
        else:
            base = "i32" if ty == "s32" else "u32"
            m = {"<": (f"v_cmp_lt_{base}_e32", False),
                 "<=": (f"v_cmp_le_{base}_e32", False) if base == "u32"
                       else (f"v_cmp_gt_{base}_e32", True),
                 ">": (f"v_cmp_gt_{base}_e32", False),
                 ">=": (f"v_cmp_le_{base}_e32", True) if base == "u32"
                       else (f"v_cmp_ge_{base}_e32", False),
                 "==": (f"v_cmp_eq_{base}_e32", False),
                 "!=": (f"v_cmp_ne_{base}_e32", False)}
        mn, swap = m[op]
        self.emit(f"{mn} vcc, v{bv if swap else av}, v{av if swap else bv}")
        return Val("vpred", None, "bool")

    def call(self, node: ast.Call) -> Val:
        if not isinstance(node.func, ast.Name):
            raise CompileError("只支持简单函数调用")
        n = node.func.id
        if n == "gid":
            return self._gid()
        if n == "tid":
            r = self._alloc_var_v()
            self.emit(f"v_mov_b32_e32 v{r}, v0")
            return Val("v", r, "u32")
        if n == "bid":
            r = self._alloc_var_s()
            self.emit(f"s_mov_b32 s{r}, s6")
            return Val("s", r, "u32")
        if n == "lane":
            r = self._alloc_var_v()
            self.emit(f"v_mov_b32_e32 v{r}, v0")
            self.emit(f"v_and_b32_e32 v{r}, 63, v{r}")
            return Val("v", r, "u32")
        if n in ("f32", "u32", "s32"):
            v = self.expr(node.args[0])
            src = self.as_vreg(v)
            r = self.alloc_tmp_v()
            if n == "f32":
                self.emit(f"v_cvt_f32_u32_e32 v{r}, v{src}" if v.ty != "s32"
                          else f"v_cvt_f32_i32_e32 v{r}, v{src}")
            elif n == "u32":
                self.emit(f"v_cvt_u32_f32_e32 v{r}, v{src}")
            else:
                self.emit(f"v_cvt_i32_f32_e32 v{r}, v{src}")
            return Val("v", r, n)
        if n == "exp":
            v = self.as_vreg(self.expr(node.args[0]))
            log2e = self.alloc_tmp_v()
            bits = struct.unpack("<I", struct.pack("<f", 1.4426950408889634))[0]
            self.emit(f"v_mov_b32_e32 v{log2e}, 0x{bits:08x}")
            r = self.alloc_tmp_v()
            self.emit(f"v_mul_f32_e32 v{r}, v{v}, v{log2e}")
            self.emit("s_nop 0")          # v_exp_f32 的输入 hazard
            self.emit(f"v_exp_f32_e32 v{r}, v{r}")
            self.free_tmp_v(2)
            return Val("v", r, "f32")
        if n in ("sqrt", "rsqrt"):
            v = self.as_vreg(self.expr(node.args[0]))
            r = self.alloc_tmp_v()
            op = {"sqrt": "v_sqrt_f32_e32", "rsqrt": "v_rsq_f32_e32"}[n]
            self.emit(f"{op} v{r}, v{v}")
            return Val("v", r, "f32")
        if n == "fma":
            a, b, c = (self.as_vreg(self.expr(x)) for x in node.args)
            r = self.alloc_tmp_v()
            self.emit(f"v_fma_f32 v{r}, v{a}, v{b}, v{c}")
            return Val("v", r, "f32")
        if n == "fabs":
            v = self.as_vreg(self.expr(node.args[0]))
            r = self.alloc_tmp_v()
            self.emit(f"v_and_b32_e32 v{r}, 0x7fffffff, v{v}")
            return Val("v", r, "f32")
        if n == "max":
            a, b = (self.as_vreg(self.expr(x)) for x in node.args)
            r = self.alloc_tmp_v()
            self.emit(f"v_max_f32_e32 v{r}, v{a}, v{b}")
            return Val("v", r, "f32")
        raise CompileError(f"未知内建 {n}")

    def _gid(self) -> Val:
        # gid = blockIdx.x * group_size_x + tid
        s = self._alloc_tmp_s()
        gsoff = self.hidden_off + 12  # block_count x/y/z 之后是 group_size_x
        self.emit(f"s_load_dword s{s}, s[4:5], 0x{gsoff:x}")
        self.emit("s_waitcnt lgkmcnt(0)")
        self.emit(f"s_and_b32 s{s}, s{s}, 0xffff")
        r = self._alloc_var_v()
        b = self._alloc_var_v()
        self.emit(f"v_mov_b32_e32 v{b}, s{s}")
        self.emit(f"v_mov_b32_e32 v{r}, s6")
        self.emit(f"v_mul_lo_u32 v{r}, v{r}, v{b}")
        self.emit(f"v_add_u32_e32 v{r}, v{r}, v0")
        return Val("v", r, "u32")

    # ---------------- 内存 ----------------
    def _addr(self, ptr: Val, index: Val) -> tuple[int, int]:
        if ptr.kind != "ptr":
            raise CompileError("索引的基址不是指针")
        idx = self.as_vreg(index)
        lo, hi = self.alloc_addr_pair()
        off = self.alloc_tmp_v()
        self.emit(f"v_lshlrev_b32_e32 v{off}, 2, v{idx}")
        self.emit(f"v_mov_b32_e32 v{lo}, s{ptr.reg}")
        self.emit(f"v_mov_b32_e32 v{hi}, s{ptr.reg + 1}")
        self.emit(f"v_add_co_u32_e32 v{lo}, vcc, v{lo}, v{off}")
        self.emit(f"v_addc_co_u32_e32 v{hi}, vcc, v{hi}, v{self.zero_v}, vcc")
        self.free_tmp_v()
        return lo, hi

    def load(self, node: ast.Subscript) -> Val:
        if not isinstance(node.value, ast.Name):
            raise CompileError("只支持 base[index]")
        ptr = self.env.get(node.value.id)
        if ptr is None or ptr.kind != "ptr":
            raise CompileError(f"{node.value.id} 不是指针")
        idx = self.expr(node.slice)
        et = ptr.ty.split(":", 1)[1]
        lo, hi = self._addr(ptr, idx)
        r = self.alloc_tmp_v()
        self.emit(f"global_load_dword v{r}, v[{lo}:{hi}], off")
        self.emit("s_waitcnt vmcnt(0)")
        self.free_addr_pair()
        return Val("v", r, et)

    def store(self, node: ast.Subscript, value: Val) -> None:
        if not isinstance(node.value, ast.Name):
            raise CompileError("只支持 base[index]")
        ptr = self.env.get(node.value.id)
        if ptr is None or ptr.kind != "ptr":
            raise CompileError(f"{node.value.id} 不是指针")
        idx = self.expr(node.slice)
        src = self.as_vreg(value)
        lo, hi = self._addr(ptr, idx)
        self.emit(f"global_store_dword v[{lo}:{hi}], v{src}, off")
        self.emit("s_waitcnt vmcnt(0)")
        self.free_addr_pair()

    # ---------------- 语句 ----------------
    def stmt(self, node) -> None:
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1:
                raise CompileError("只支持单赋值")
            target = node.targets[0]
            val = self.expr(node.value)
            if isinstance(target, ast.Name):
                self._assign_name(target.id, val)
            elif isinstance(target, ast.Subscript):
                self.store(target, val)
            else:
                raise CompileError("不支持的赋值目标")
        elif isinstance(node, ast.AnnAssign):
            if not isinstance(node.target, ast.Name):
                raise CompileError("只支持 Name 注解")
            ty, is_ptr = parse_type(_ty_name(node.annotation))
            if is_ptr:
                raise CompileError("局部指针不支持")
            if node.value is None:
                raise CompileError("局部变量需要初值")
            val = self.expr(node.value)
            self._assign_name(node.target.id, val, force_ty=ty)
        elif isinstance(node, ast.AugAssign):
            t = node.target
            if not isinstance(t, ast.Name):
                raise CompileError("只支持 Name 增强赋值")
            cur = self.env[t.id]
            rhs = self.expr(node.value)
            op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}[type(node.op)]
            self._assign_name(t.id, self.binop(op, cur, rhs))
        elif isinstance(node, ast.If):
            self._if(node)
        elif isinstance(node, ast.For):
            self._for(node)
        elif isinstance(node, ast.Break):
            self.emit(f"s_branch {self.loop_stack[-1][1]}")
        elif isinstance(node, ast.Continue):
            self.emit(f"s_branch {self.loop_stack[-1][0]}")
        elif isinstance(node, ast.Pass):
            return
        else:
            raise CompileError(f"不支持的语句 {type(node).__name__}")

    def _assign_name(self, name: str, val: Val, force_ty: str | None = None) -> None:
        ty = force_ty or val.ty
        if name in self.env:
            dst = self.env[name]
            if dst.kind == "v":
                self.emit(f"v_mov_b32_e32 v{dst.reg}, v{self.as_vreg(val)}")
            elif dst.kind == "s":
                if self.is_uniform(val):
                    self.emit(f"s_mov_b32 s{dst.reg}, s{self.as_sreg(val)}")
                else:
                    # 不把 varying 收回 uniform
                    r = self._alloc_var_v()
                    self.emit(f"v_mov_b32_e32 v{r}, v{self.as_vreg(val)}")
                    self.env[name] = Val("v", r, ty)
            else:
                raise CompileError("不能赋值给指针")
            return
        if self.is_uniform(val) and ty != "f32":
            r = self._alloc_var_s()
            self.emit(f"s_mov_b32 s{r}, s{self.as_sreg(val)}")
            self.env[name] = Val("s", r, ty)
        else:
            r = self._alloc_var_v()
            self.emit(f"v_mov_b32_e32 v{r}, v{self.as_vreg(val)}")
            self.env[name] = Val("v", r, ty)

    def _if(self, node: ast.If) -> None:
        cond = self.expr(node.test)
        if cond.kind == "vpred":
            if node.orelse:
                raise CompileError("varying if/else 暂不支持（改成无 else 或 uniform 条件）")
            save = self.save_s
            self.save_s += 2
            end = self.new_label("if_end")
            self.emit(f"s_and_saveexec_b64 s[{save}:{save + 1}], vcc")
            self.emit(f"s_cbranch_execz {end}")
            for st in node.body:
                self.stmt(st)
            self.emit(f"s_or_b64 exec, exec, s[{save}:{save + 1}]")
            self.label(end)
            self.save_s -= 2
            return
        end = self.new_label("if_end")
        els = self.new_label("if_else")
        self.emit(f"s_cbranch_scc0 {els if node.orelse else end}")
        for st in node.body:
            self.stmt(st)
        if node.orelse:
            self.emit(f"s_branch {end}")
            self.label(els)
            for st in node.orelse:
                self.stmt(st)
        self.label(end)

    def _for(self, node: ast.For) -> None:
        if not isinstance(node.target, ast.Name) or not isinstance(node.iter, ast.Call) \
                or not isinstance(node.iter.func, ast.Name) or node.iter.func.id != "range":
            raise CompileError("只支持 for i in range(...)")
        args = [self.expr(a) for a in node.iter.args]
        if len(args) == 1:
            start, endv = Val("lit", None, "u32", 0), args[0]
        elif len(args) == 2:
            start, endv = args
        else:
            raise CompileError("range 只支持 1-2 个参数")
        if not self.is_uniform(endv) or not self.is_uniform(start):
            raise CompileError("for range 的边界必须是 uniform int")
        s_end = self.as_sreg(endv)
        s_i = self._alloc_var_s()
        if start.kind == "lit":
            self.emit(f"s_mov_b32 s{s_i}, {int(start.value)}")
        else:
            self.emit(f"s_mov_b32 s{s_i}, s{self.as_sreg(start)}")
        loop = self.new_label("for_loop")
        inc = self.new_label("for_inc")
        end = self.new_label("for_end")
        self.label(loop)
        self.emit(f"s_cmp_lt_u32 s{s_i}, s{s_end}")
        self.emit(f"s_cbranch_scc0 {end}")
        self.env[node.target.id] = Val("s", s_i, "u32")
        self.loop_stack.append((inc, end))
        for st in node.body:
            self.stmt(st)
        self.loop_stack.pop()
        self.label(inc)
        self.emit(f"s_add_i32 s{s_i}, s{s_i}, 1")
        self.emit(f"s_branch {loop}")
        self.label(end)

    # ---------------- ABI ----------------
    def hidden_args(self) -> list[tuple[str, int, int]]:
        o = self.hidden_off
        out = [("hidden_block_count_x", o, 4), ("hidden_block_count_y", o + 4, 4),
               ("hidden_block_count_z", o + 8, 4), ("hidden_group_size_x", o + 12, 2),
               ("hidden_group_size_y", o + 14, 2), ("hidden_group_size_z", o + 16, 2),
               ("hidden_remainder_x", o + 18, 2), ("hidden_remainder_y", o + 20, 2),
               ("hidden_remainder_z", o + 22, 2)]
        o2 = _align(o + 24, 8)
        out += [("hidden_global_offset_x", o2, 8), ("hidden_global_offset_y", o2 + 8, 8),
                ("hidden_global_offset_z", o2 + 16, 8), ("hidden_grid_dims", o2 + 24, 2)]
        return out

    def arg_meta(self) -> tuple[list[dict], int]:
        args = []
        for p in self.params:
            args.append({".offset": p.off, ".size": 8 if p.is_ptr else 4,
                         ".address_space": "global" if p.is_ptr else None,
                         ".value_kind": "global_buffer" if p.is_ptr else "by_value"})
        for name, off, size in self.hidden_args():
            args.append({".offset": off, ".size": size, ".value_kind": name})
        args = [{k: v for k, v in a.items() if v is not None} for a in args]
        hidden_end = max(off + size for _n, off, size in self.hidden_args())
        ksize = _align(hidden_end, 8)
        return args, ksize

    @property
    def vgpr_count(self) -> int:
        return max(self.max_v + 1, 4)

    @property
    def sgpr_count(self) -> int:
        return max(self.max_s + 1, 8)


def compile_source(source: str, out_dir: pathlib.Path, only: str | None = None) -> list[pathlib.Path]:
    tree = ast.parse(source)
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if only:
        fns = [f for f in fns if f.name == only]
    if not fns:
        raise CompileError("没有找到内核函数")
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for fn in fns:
        cg = CodeGen(fn)
        text = cg.compile()
        s_path = out_dir / f"{fn.name}.s"
        s_path.write_text(text, encoding="utf-8")
        code, _ins, _lbl = asm.assemble(s_path)
        args, ksize = cg.arg_meta()
        kernel = {
            "name": fn.name, "code": code, "args": args,
            "kernarg_size": ksize, "kernarg_align": 8,
            "sgpr_count": cg.sgpr_count, "vgpr_count": cg.vgpr_count,
            "group_segment": 0, "private_segment": 0,
        }
        elf = build_elf([kernel])
        h_path = out_dir / f"{fn.name}.hsaco"
        h_path.write_bytes(elf)
        outputs.append(h_path)
    return outputs


def compile_file(path: pathlib.Path, out_dir: pathlib.Path, only: str | None = None):
    return compile_source(path.read_text(encoding="utf-8"), out_dir, only)
