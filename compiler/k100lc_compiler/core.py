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


ELEM_SIZE = {"f32": 4, "u32": 4, "s32": 4, "u16": 2, "s16": 2, "u8": 1, "s8": 1}


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
    # 这条值是一次 global_load 的结果、还没等过 `s_waitcnt`（延迟等待：同一个
    # 语句里连续几个 load 可以同时在飞，第一次用到时才排空）
    pending: bool = False


class CodeGen:
    def __init__(self, fn: ast.FunctionDef):
        self.fn = fn
        self.lines: list[str] = []
        self.params: list[Param] = []
        self.env: dict[str, Val] = {}
        self.var_v = 2
        self.var_regs: dict[str, int] = {}      # 命名变量 → 专属 VGPR
        self.var_anon = 2                       # 匿名变量（gid/tid/lane）游标
        self.var_anon_end = 2
        self.var_s = 16
        self.tmp_v = 64
        self.tmp_base = 64
        # 临时寄存器池：`alloc_tmp_v` 优先从 `free_temps` 取，`_release` 在用完
        # 之后归还。没有这个池的时候临时值是单调递增的——`x[0]+x[1]+…+x[n]`
        # 这种长表达式 60 项就把 v245 用光了（其实同时只有两三个是活的）。
        self.live_temps: set[int] = set()
        self.free_temps: list[int] = []
        self.new_temps: list[int] = []        # as_vreg 为常量/标量新建的临时
        self.pending_vals: list[Val] = []     # 发出去还没排空的 load 结果
        self.tmp_s = 64
        self.addr_v = 200
        self.save_s = 48
        self.max_v = 1
        self.max_s = 15
        self.loop_stack: list[tuple[str, str]] = []
        # 当前嵌套在哪些「varying 条件的 if」里（存每个区的编号，最内层在末尾）。
        # 用途一：`break`/`continue` 在区里是错的——标量分支会让**所有** lane
        # 一起跳出/跳回，而不是只影响条件成立的那些 lane。
        # 用途二：标量（SGPR）变量只在它**自己那个区**里才是整波一致的记账；
        # 换个区赋值就说明要的是 per-lane 语义（见 `_assign_name`）。
        self.varying_regions: list[int] = []
        self.region_n = 0
        self.var_region: dict[str, int | None] = {}   # 变量是在哪个 varying 区里建的
        self.zero_v = 1
        self.label_n = 0
        self._parse_params()
        self._plan_registers()

    # ---------------- 基础 ----------------
    def emit(self, text: str) -> None:
        self.lines.append(text)

    def label(self, name: str) -> None:
        self.lines.append(f"{name}:")

    def new_label(self, prefix: str) -> str:
        self.label_n += 1
        return f"{prefix}_{self.label_n}"

    def alloc_tmp_v(self) -> int:
        if self.free_temps:
            r = self.free_temps.pop()
        else:
            r = self.tmp_v
            self.tmp_v += 1
            if self.tmp_v > 245:
                raise CompileError("VGPR 溢出（不做 spill）：同时存活的临时值太多")
        self.max_v = max(self.max_v, r)
        self.live_temps.add(r)
        return r

    def _plan_registers(self) -> None:
        """预扫描一遍 AST，一次把「变量区」和「临时区」的边界定下来。

        以前两边都从 v64 附近往上涨：局部变量超过 ~62 个时变量区和临时区会
        **重叠**——同一个寄存器既放变量又当临时值，生成的内核静默算错
        （实测 62 个局部变量就开始出错）。现在：

          * 每个命名变量拿一个专属 VGPR（v2 起，按首次出现顺序）；
          * `gid()/tid()/lane()` 的匿名变量按调用点预留；
          * 临时值从 `max(64, 变量区末尾)` 起，和变量区完全不重叠。
        """
        names: list[str] = []
        seen: set[str] = set()
        anon = 0

        def take(nm: str) -> None:
            if nm not in seen:
                seen.add(nm)
                names.append(nm)

        for node in ast.walk(self.fn):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        take(t.id)
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    take(node.target.id)
            elif isinstance(node, ast.AugAssign):
                if isinstance(node.target, ast.Name):
                    take(node.target.id)
            elif isinstance(node, ast.For):
                tgt = getattr(node.target, "id", None)
                if tgt:
                    take(tgt)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ("tid", "lane"):
                    anon += 1
                elif node.func.id == "gid":
                    anon += 2

        r = 2
        for nm in names:
            self.var_regs[nm] = r
            r += 1
        self.var_anon = r
        self.var_anon_end = r + anon
        # 地址对（lo/hi）单独占两个连续的寄存器，插在变量区和临时区之间。
        # 以前写死在 v254/v255，导致**每个**带访存的编译内核都声明 256 个 VGPR
        # （占用率只有实际需要的 1/3）；现在按实际用量收口。
        self.addr_lo = self.var_anon_end
        self.addr_hi = self.var_anon_end + 1
        self.var_v = self.addr_hi + 1
        self.tmp_base = max(64, self.addr_hi + 1)
        self.tmp_v = self.tmp_base
        self.max_v = max(self.max_v, self.var_anon_end - 1)
        if self.tmp_base > 200:
            raise CompileError(
                f"{self.fn.name}: 变量太多，VGPR 变量区已到 v{self.var_anon_end}"
                f"（上限 v200）")

    def free_tmp_v(self, reg: int) -> None:
        """归还一个临时寄存器（只有由 `alloc_tmp_v` 真正发出去过的才收）。"""
        if reg in self.live_temps:
            self.live_temps.discard(reg)
            self.free_temps.append(reg)

    def _release(self, *vals) -> None:
        """一条指令发完之后调用：归还它的操作数临时值，以及 `as_vreg` 为
        常量/标量新建的临时值。调用方保证这些值没有别的引用。
        """
        for v in vals:
            if v is not None and v.kind == "v" and v.reg is not None:
                self.free_tmp_v(v.reg)
        for r in self.new_temps:
            self.free_tmp_v(r)
        self.new_temps.clear()

    def _cur_region(self) -> int | None:
        """当前最内层 varying 区的编号（不在任何区里就是 None）。"""
        return self.varying_regions[-1] if self.varying_regions else None

    def _region_of(self, name: str) -> int | None:
        """变量是在哪个 varying 区里建的。"""
        return self.var_region.get(name)

    def alloc_addr_pair(self) -> tuple[int, int]:
        # 所有 load/store 复用同一个地址对；表达式求值不会同时持有两个地址
        # （前一个 load 的结果已经在临时 VGPR 里）。这样临时寄存器不会撞地址对。
        self.max_v = max(self.max_v, self.addr_hi)
        return self.addr_lo, self.addr_hi

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
        """匿名变量（`gid()` / `tid()` / `lane()` 用）——预算在预扫描里定死。"""
        r = self.var_anon
        self.var_anon += 1
        if self.var_anon > self.var_anon_end:
            raise CompileError("匿名 VGPR 预算耗尽（编译器内部错误）")
        self.max_v = max(self.max_v, r)
        return r

    def _named_vreg(self, name: str) -> int:
        """命名变量专属的 VGPR（预扫描时分配，不会和临时值撞）。"""
        r = self.var_regs.get(name)
        if r is None:
            raise CompileError(f"{name}: 没有预留 VGPR（编译器内部错误）")
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

    def _alloc_tmp_s(self, size: int = 1) -> int:
        r = self.tmp_s
        if size == 2 and r % 2:
            r += 1                       # s[x:x+1] 必须偶数对齐
        self.tmp_s = r + size
        if self.tmp_s > 101:
            raise CompileError("临时 SGPR 超过 101")
        self.max_s = max(self.max_s, r + size - 1)
        return r

    # ---------------- 值与类型 ----------------
    def is_uniform(self, v: Val) -> bool:
        return v.kind in ("s", "lit") and v.ty != "f32"

    def _f32_operand(self, v: Val) -> Val:
        """f32 上下文里的整数常量要**当成数值**转成 f32。

        `as_vreg` 对整型字面量发的是 `v_mov_b32 vN, <整数>`——那是位模式，
        在浮点运算里等于把 1 变成 1.4e-45。所以 `x[i] + 1`、`x[i] > 1`、
        `y[i] = 1` 这些写法都会静默算错（v1.8.1 之前一直如此）。
        """
        if v.kind == "lit" and v.ty != "f32" and not isinstance(v.value, float):
            return Val("lit", None, "f32", float(v.value))
        if v.ty not in ("f32", "bool") and v.kind in ("v", "s"):
            # 整数**变量**同理：`m = 3` 之后 `x[i] * m` 也要先 cvt 成 f32
            src = self.as_vreg(v)
            r = self.alloc_tmp_v()
            op = "v_cvt_f32_i32_e32" if v.ty == "s32" else "v_cvt_f32_u32_e32"
            self.emit(f"{op} v{r}, v{src}")
            self._release(v)
            return Val("v", r, "f32")
        return v

    def as_vreg(self, v: Val) -> int:
        if v.pending:
            self._wait_loads()
        if v.kind == "v":
            return v.reg
        r = self.alloc_tmp_v()
        self.new_temps.append(r)     # 没有 Val 持有它，用完由 _release 归还
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
        if v.pending:
            self._wait_loads()
        if v.kind == "s":
            return v.reg
        if v.kind == "lit":
            r = self._alloc_tmp_s()
            self.emit(f"s_mov_b32 s{r}, {int(v.value)}")
            return r
        raise CompileError(f"不是 uniform 值: {v}")

    def _wait_loads(self) -> None:
        """把所有还在飞的 global_load 排空（`s_waitcnt vmcnt(0)`）。

        一次 `vmcnt(0)` 会把**所有**未完成的 VMEM 载入都等到，所以这里可以
        一次性把所有 pending 标清掉——这正是「延迟等待」能提升访存并行度的
        原因：同一个语句里连着发的几条 load 会一起在飞。
        """
        if not self.pending_vals:
            return
        self.emit("s_waitcnt vmcnt(0)")
        for pv in self.pending_vals:
            pv.pending = False
        self.pending_vals.clear()

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
                if n in ("exp", "sqrt", "rsqrt", "fma", "fabs", "floor", "ceil",
                         "trunc", "rint", "fract", "ubyte"):
                    return "f32"
                if n in ("max", "min"):
                    t0 = self.type_of(node.args[0]) if node.args else "f32"
                    t1 = self.type_of(node.args[1]) if len(node.args) > 1 else "f32"
                    if "f32" in (t0, t1):
                        return "f32"
                    return "s32" if "s32" in (t0, t1) else "u32"
                if n in ("gid", "tid", "bid", "lane"):
                    return "u32"
                if n in ("f32", "u32", "s32"):
                    return n
                if n == "load16":
                    return "u32"
                if n == "f16_to_f32":
                    return "f32"
                if n == "s8":
                    return "s32"
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
                  ast.Mod: "%",
                  ast.LShift: "<<", ast.RShift: ">>",
                  ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^"}.get(type(node.op))
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
        if isinstance(node, ast.BoolOp):
            # 逻辑与/或。DSL 表达式没有副作用，所以**两边都求值**、把各自的掩码
            # 在 vcc 里按位合并（不是短路求值，但结果一样）。
            # 每个比较都会写 vcc，所以合并前先把当前掩码存进一对 SGPR。
            if not isinstance(node.op, (ast.And, ast.Or)):
                raise CompileError("只支持 and / or")
            mn = "s_and_b64" if isinstance(node.op, ast.And) else "s_or_b64"
            acc = self._alloc_tmp_s(2)
            all_uniform = True
            first = True
            for sub in node.values:
                v = self.expr(sub)
                if v.kind == "vpred":
                    all_uniform = False
                    pass                       # 掩码已经在 vcc 里
                elif v.kind == "spred":
                    # 标量条件在 SCC 里，展成 64 位掩码再合并
                    self.emit("s_cselect_b64 vcc, -1, 0")
                else:
                    raise CompileError("and/or 的操作数必须是比较结果")
                if first:
                    self.emit(f"s_mov_b64 s[{acc}:{acc + 1}], vcc")
                    first = False
                else:
                    self.emit(f"{mn} vcc, vcc, s[{acc}:{acc + 1}]")
                    self.emit(f"s_mov_b64 s[{acc}:{acc + 1}], vcc")
            self.emit(f"s_mov_b64 vcc, s[{acc}:{acc + 1}]")
            if all_uniform:
                # 全部是标量条件 → 结果也该是标量（否则 `while a and b` 这种
                # 全 uniform 的写法会被 `_while` 当成 varying 拒掉）
                self.emit(f"s_cmp_lg_u32 s{acc}, 0")
                return Val("spred", None, "bool")
            return Val("vpred", None, "bool")
        if isinstance(node, ast.Subscript):
            return self.load(node)
        if isinstance(node, ast.Call):
            return self.call(node)
        raise CompileError(f"不支持的表达式 {ast.dump(node)}")

    def binop(self, op: str, a: Val, b: Val, ty: str | None = None) -> Val:
        ty = ty or (a.ty if a.ty != "u32" else b.ty)
        if ty == "f32":
            a, b = self._f32_operand(a), self._f32_operand(b)
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
                    self.free_tmp_v(t)
            else:
                raise CompileError(f"f32 不支持 {op}")
            self._release(a, b)
            return Val("v", dst, "f32")
        if self.is_uniform(a) and self.is_uniform(b):
            sa, sb = self.as_sreg(a), self.as_sreg(b)
            dst = self._alloc_tmp_s()
            m = {"+": "s_add_u32", "-": "s_sub_u32", "&": "s_and_b32",
                 "|": "s_or_b32", "^": "s_xor_b32", "*": "s_mul_i32"}
            if op == "<<":
                self.emit(f"s_lshl_b32 s{dst}, s{sa}, s{sb}")
            elif op == ">>":
                self.emit(f"{'s_ashr_i32' if ty == 's32' else 's_lshr_b32'} "
                          f"s{dst}, s{sa}, s{sb}")
            elif op in ("/", "%"):
                # 与整数 varying 路径同约定：只支持 2 的幂常量除数
                d = int(b.value) if b.kind == "lit" else None
                if d is None or d <= 0 or (d & (d - 1)) != 0 or ty == "s32":
                    raise CompileError(
                        f"uniform int {op} 只支持 2 的幂常量除数（收到 {b!r}）")
                if op == "/":
                    self.emit(f"s_lshr_b32 s{dst}, s{sa}, {d.bit_length() - 1}")
                else:
                    self.emit(f"s_and_b32 s{dst}, s{sa}, {d - 1}")
            elif op not in m:
                raise CompileError(f"uniform 不支持 {op}")
            else:
                self.emit(f"{m[op]} s{dst}, s{sa}, s{sb}")
            return Val("s", dst, ty)
        av, bv = self.as_vreg(a), self.as_vreg(b)
        dst = self.alloc_tmp_v()
        m = {"+": "v_add_u32_e32", "-": "v_sub_u32_e32", "*": "v_mul_lo_u32",
             "&": "v_and_b32_e32", "|": "v_or_b32_e32", "^": "v_xor_b32_e32"}
        # 整数除 / 取模：只支持「2 的幂的常量除数」，用移位和掩码精确实现。
        # 一般的整数除法要么上 magic 数（需要知道取值范围，不然会静默算错），
        # 要么上完整除法序列，这里先不做——需要时用 `>>`/`&` 自己写。
        if op in ("/", "%"):
            d = int(b.value) if b.kind == "lit" else None
            if d is None or d <= 0 or (d & (d - 1)) != 0:
                raise CompileError(
                    f"int {op} 只支持 2 的幂常量除数（收到 {b!r}）；"
                    f"其它除数请用 `>>`/`&` 展开，或先在主机侧算好")
            if ty == "s32":
                raise CompileError("int 除/取模暂不支持 s32（负数移位语义不同）")
            shift = d.bit_length() - 1
            if op == "/":
                self.emit(f"v_lshrrev_b32_e32 v{dst}, {shift}, v{av}")
            else:
                self.emit(f"v_and_b32_e32 v{dst}, {d - 1}, v{av}")
        elif op == "<<":
            self.emit(f"v_lshlrev_b32_e32 v{dst}, v{bv}, v{av}")
        elif op == ">>":
            self.emit(f"{'v_ashrrev_i32_e32' if ty == 's32' else 'v_lshrrev_b32_e32'} "
                      f"v{dst}, v{bv}, v{av}")
        elif op in m:
            self.emit(f"{m[op]} v{dst}, v{av}, v{bv}")
        else:
            raise CompileError(f"int 不支持 {op}")
        self._release(a, b)
        return Val("v", dst, ty)

    def compare(self, op: str, a: Val, b: Val) -> Val:
        ty = "f32" if "f32" in (a.ty, b.ty) else ("s32" if "s32" in (a.ty, b.ty) else "u32")
        if self.is_uniform(a) and self.is_uniform(b) and ty != "f32":
            sa, sb = self.as_sreg(a), self.as_sreg(b)
            if op in ("==", "!="):
                # 两个补码下「相等」与符号无关，统一用 u32 形式
                # （编码表里没有 s_cmp_eq_i32 / s_cmp_lg_i32）
                mn = "s_cmp_eq_u32" if op == "==" else "s_cmp_lg_u32"
                self.emit(f"{mn} s{sa}, s{sb}")
                return Val("spred", None, "bool")
            base = "i32" if ty == "s32" else "u32"
            m = {"<": ("s_cmp_lt", False), "<=": ("s_cmp_ge", True),
                 ">": ("s_cmp_gt", False), ">=": ("s_cmp_ge", False)}
            mn, swap = m[op]
            self.emit(f"{mn}_{base} s{sb if swap else sa}, s{sa if swap else sb}")
            return Val("spred", None, "bool")
        if ty == "f32":
            a, b = self._f32_operand(a), self._f32_operand(b)
        av, bv = self.as_vreg(a), self.as_vreg(b)
        if ty == "f32":
            # f32 比较只有 e64 形式支持「两个都是 VGPR」：e32 的第一个源在
            # 编码表里是 ssrc（标量/内联常量），`v_cmp_gt_f32_e32 vcc, v3, v64`
            # 汇编器会直接拒绝（v1.6.2 之前所有 f32 比较都编不过）。
            # 统一用 `v_cmp_lt_f32_e64 vcc, x, y`，再靠交换操作数 / 取反掩码
            # 拼出其余运算符。注意：取反得到的 `>=`/`<=` 是「不小于」，
            # NaN 时与有序比较不同（LLVM 的 ordered 语义）。
            self._cmp_f32(NodeOp=op, av=av, bv=bv)
            self._release(a, b)
            return Val("vpred", None, "bool")
        self._cmp_int(ty, op, av, bv)
        self._release(a, b)
        return Val("vpred", None, "bool")

    def _cmp_int(self, ty: str, op: str, av: int, bv: int) -> None:
        """整数比较结果放进 vcc。

        编码表里两个源都能是 VGPR 的整数比较只有三条：
        `v_cmp_lt_u32_e64` / `v_cmp_gt_i32_e64` / `v_cmp_eq_u32_e32`。
        其余运算符用「交换操作数 + 取反掩码」拼出来（`s_xor_b64 vcc, vcc, -1`）。
        v1.6.4 之前 u32 的 `<= >= !=`、s32 的 `<= > >= == !=` 都编不过。
        """
        if op in ("==", "!="):
            # 相等与符号无关，直接用 u32 形式
            self.emit(f"v_cmp_eq_u32_e32 vcc, v{av}, v{bv}")
            if op == "!=":
                self.emit("s_xor_b64 vcc, vcc, -1")
            return
        if op in (">", "<="):           # a>b ⇔ b<a；a<=b ⇔ !(b<a)
            av, bv = bv, av
        if ty == "u32":
            self.emit(f"v_cmp_lt_u32_e64 vcc, v{av}, v{bv}")
        else:
            self.emit(f"v_cmp_gt_i32_e64 vcc, v{bv}, v{av}")   # y>x ⇔ x<y
        if op in ("<=", ">="):
            self.emit("s_xor_b64 vcc, vcc, -1")

    def _cmp_f32(self, NodeOp: str, av: int, bv: int) -> None:
        """把 f32 比较结果放进 vcc（只用 `v_cmp_lt_f32_e64`）。

        `<` 直接比；`>` 交换操作数；`<=`/`>=` 再用 `s_xor_b64` 取反；
        `==` 用「a>=b 且 b>=a」，`!=` 再取反一次。
        """
        op = NodeOp
        negate = op in ("<=", ">=")
        if op in (">", "<="):            # a>b ⇔ b<a；a<=b ⇔ !(b<a)
            av, bv = bv, av
        if op in ("==", "!="):
            p = self._alloc_tmp_s(2)
            self.emit(f"v_cmp_lt_f32_e64 s[{p}:{p + 1}], v{av}, v{bv}")
            self.emit(f"s_xor_b64 s[{p}:{p + 1}], s[{p}:{p + 1}], -1")   # a >= b
            self.emit(f"v_cmp_lt_f32_e64 vcc, v{bv}, v{av}")
            self.emit("s_xor_b64 vcc, vcc, -1")                          # b >= a
            self.emit(f"s_and_b64 vcc, vcc, s[{p}:{p + 1}]")
            if op == "!=":
                self.emit("s_xor_b64 vcc, vcc, -1")
            return
        self.emit(f"v_cmp_lt_f32_e64 vcc, v{av}, v{bv}")
        if negate:
            self.emit("s_xor_b64 vcc, vcc, -1")

    def call(self, node: ast.Call) -> Val:
        if not isinstance(node.func, ast.Name):
            raise CompileError("只支持简单函数调用")
        n = node.func.id
        if n == "gid":
            return self._gid()
        if n == "load16":
            ptr = self.expr(node.args[0])
            if ptr.kind != "ptr":
                raise CompileError("load16 第一个参数必须是指针")
            idx = self.expr(node.args[1])
            off = Val("v", self.as_vreg(idx), "u32")
            lo, hi = self._addr(ptr, off, "u8")
            r = self.alloc_tmp_v()
            self.emit(f"global_load_ushort v{r}, v[{lo}:{hi}], off")
            self.emit("s_waitcnt vmcnt(0)")
            self.free_addr_pair()
            self._release(idx)
            return Val("v", r, "u32")
        if n == "f16_to_f32":
            arg = self.expr(node.args[0])
            v = self.as_vreg(arg)
            r = self.alloc_tmp_v()
            self.emit(f"v_cvt_f32_f16_e32 v{r}, v{v}")
            self._release(arg)
            return Val("v", r, "f32")
        if n == "s8":
            arg = self.expr(node.args[0])
            v = self.as_vreg(arg)
            r = self.alloc_tmp_v()
            self.emit(f"v_lshlrev_b32_e32 v{r}, 24, v{v}")
            self.emit(f"v_ashrrev_i32_e32 v{r}, 24, v{r}")
            self._release(arg)
            return Val("v", r, "s32")
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
            if n in ("u32", "s32") and v.ty in ("u8", "s8", "u16", "s16", "u32", "s32"):
                # 同宽整数只是类型视图，位模式不变
                return Val("v", self.as_vreg(v), n)
            src = self.as_vreg(v)
            r = self.alloc_tmp_v()
            if n == "f32":
                self.emit(f"v_cvt_f32_u32_e32 v{r}, v{src}" if v.ty != "s32"
                          else f"v_cvt_f32_i32_e32 v{r}, v{src}")
            elif n == "u32":
                self.emit(f"v_cvt_u32_f32_e32 v{r}, v{src}")
            else:
                self.emit(f"v_cvt_i32_f32_e32 v{r}, v{src}")
            self._release(v)
            return Val("v", r, n)
        if n == "exp":
            arg = self.expr(node.args[0])
            arg = self._f32_operand(arg)
            v = self.as_vreg(arg)
            log2e = self.alloc_tmp_v()
            bits = struct.unpack("<I", struct.pack("<f", 1.4426950408889634))[0]
            self.emit(f"v_mov_b32_e32 v{log2e}, 0x{bits:08x}")
            r = self.alloc_tmp_v()
            self.emit(f"v_mul_f32_e32 v{r}, v{v}, v{log2e}")
            self.emit("s_nop 0")          # v_exp_f32 的输入 hazard
            self.emit(f"v_exp_f32_e32 v{r}, v{r}")
            self.emit("s_nop 0")          # 结果的读 hazard（同 sqrt/rcp）
            self.free_tmp_v(log2e)
            self._release(arg)
            return Val("v", r, "f32")
        if n in ("sqrt", "rsqrt"):
            arg = self.expr(node.args[0])
            arg = self._f32_operand(arg)
            v = self.as_vreg(arg)
            r = self.alloc_tmp_v()
            op = {"sqrt": "v_sqrt_f32_e32", "rsqrt": "v_rsq_f32_e32"}[n]
            self.emit("s_nop 0")          # 源的读 hazard（与 exp 同理）
            self.emit(f"{op} v{r}, v{v}")
            # 与 `v_exp_f32` / `v_rcp_f32` 同类：结果刚写出来就被后续 VALU/VMEM
            # 读的话要隔一条 `s_nop 0`。少了它会出现**同一个 wave 里只有部分
            # lane 算错**（实测 `y[i] = sqrt(4)` 时每 16 个元素里 8~11 号是 0）。
            self.emit("s_nop 0")
            self._release(arg)
            return Val("v", r, "f32")
        if n == "fma":
            argv = [self.expr(x) for x in node.args]
            argv = [self._f32_operand(x) for x in argv]
            a, b, c = (self.as_vreg(x) for x in argv)
            r = self.alloc_tmp_v()
            self.emit(f"v_fma_f32 v{r}, v{a}, v{b}, v{c}")
            self._release(*argv)
            return Val("v", r, "f32")
        if n == "fabs":
            arg = self.expr(node.args[0])
            arg = self._f32_operand(arg)
            v = self.as_vreg(arg)
            r = self.alloc_tmp_v()
            self.emit(f"v_and_b32_e32 v{r}, 0x7fffffff, v{v}")
            self._release(arg)
            return Val("v", r, "f32")
        if n in ("max", "min"):
            argv = [self.expr(x) for x in node.args]
            if len(argv) != 2:
                raise CompileError(f"{n} 需要两个参数")
            is_f = "f32" in (argv[0].ty, argv[1].ty)
            if not is_f and "s32" in (argv[0].ty, argv[1].ty):
                raise CompileError(f"{n} 暂不支持 s32（编码表里没有 i32 的 max/min）")
            if is_f:
                argv = [self._f32_operand(x) for x in argv]
            a, b = (self.as_vreg(x) for x in argv)
            r = self.alloc_tmp_v()
            if is_f:
                if n == "max":
                    self.emit(f"v_max_f32_e32 v{r}, v{a}, v{b}")
                else:
                    # 编码表里没有 f32 的 `v_min`（只有 `v_max` / `v_max3`），
                    # 用 `-max(-a, -b)` 精确实现：浮点取负就是翻符号位。
                    na, nb = self.alloc_tmp_v(), self.alloc_tmp_v()
                    self.emit(f"v_xor_b32_e32 v{na}, 0x80000000, v{a}")
                    self.emit(f"v_xor_b32_e32 v{nb}, 0x80000000, v{b}")
                    self.emit(f"v_max_f32_e32 v{r}, v{na}, v{nb}")
                    self.emit(f"v_xor_b32_e32 v{r}, 0x80000000, v{r}")
                    self.free_tmp_v(na)
                    self.free_tmp_v(nb)
                ty = "f32"
            else:
                # 整数走无符号版本（v1.7.1 之前不管什么类型都发 v_max_f32，
                # 整数会**静默算错**）
                self.emit(f"{'v_max_u32_e32' if n == 'max' else 'v_min_u32_e32'} "
                          f"v{r}, v{a}, v{b}")
                ty = "u32"
            self._release(*argv)
            return Val("v", r, ty)
        if n in ("floor", "ceil", "trunc", "rint", "fract", "ubyte"):
            argv = [self.expr(x) for x in node.args]
            if n != "ubyte":                     # ubyte 要的就是整数
                argv = [self._f32_operand(x) for x in argv]
            v = self.as_vreg(argv[0])
            r = self.alloc_tmp_v()
            if n == "ubyte":
                # 取整数的低 8 位按无符号转 f32（解码内核常用）
                self.emit(f"v_cvt_f32_ubyte0_e32 v{r}, v{v}")
            elif n == "ceil":
                # 编码表里没有 v_ceil：ceil(x) = -floor(-x)
                t = self.alloc_tmp_v()
                self.emit(f"v_xor_b32_e32 v{t}, 0x80000000, v{v}")
                self.emit(f"v_floor_f32_e32 v{r}, v{t}")
                self.emit(f"v_xor_b32_e32 v{r}, 0x80000000, v{r}")
                self.free_tmp_v(t)
            else:
                op = {"floor": "v_floor_f32_e32", "trunc": "v_trunc_f32_e32",
                      "rint": "v_rndne_f32_e32", "fract": "v_fract_f32_e32"}[n]
                self.emit(f"{op} v{r}, v{v}")
            self._release(*argv)
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
    def _addr(self, ptr: Val, index: Val, elem_ty: str = "f32") -> tuple[int, int]:
        if ptr.kind != "ptr":
            raise CompileError("索引的基址不是指针")
        idx = self.as_vreg(index)
        lo, hi = self.alloc_addr_pair()
        off = self.alloc_tmp_v()
        esize = ELEM_SIZE.get(elem_ty, 4)
        if esize == 1:
            self.emit(f"v_mov_b32_e32 v{off}, v{idx}")
        else:
            shift = {2: 1, 4: 2}.get(esize, 2)
            self.emit(f"v_lshlrev_b32_e32 v{off}, {shift}, v{idx}")
        self.emit(f"v_mov_b32_e32 v{lo}, s{ptr.reg}")
        self.emit(f"v_mov_b32_e32 v{hi}, s{ptr.reg + 1}")
        self.emit(f"v_add_co_u32_e32 v{lo}, vcc, v{lo}, v{off}")
        self.emit(f"v_addc_co_u32_e32 v{hi}, vcc, v{hi}, v{self.zero_v}, vcc")
        self.free_tmp_v(off)
        self._release(index)
        return lo, hi

    def load(self, node: ast.Subscript) -> Val:
        if not isinstance(node.value, ast.Name):
            raise CompileError("只支持 base[index]")
        ptr = self.env.get(node.value.id)
        if ptr is None or ptr.kind != "ptr":
            raise CompileError(f"{node.value.id} 不是指针")
        idx = self.expr(node.slice)
        et = ptr.ty.split(":", 1)[1]
        lo, hi = self._addr(ptr, idx, et)
        r = self.alloc_tmp_v()
        mn = {"f32": "global_load_dword", "u32": "global_load_dword",
              "s32": "global_load_dword", "u16": "global_load_ushort",
              "s16": "global_load_ushort", "u8": "global_load_ubyte",
              "s8": "global_load_sbyte"}[et]
        self.emit(f"{mn} v{r}, v[{lo}:{hi}], off")
        self.free_addr_pair()
        # 延迟等待：先不排空，等第一次真正用到这条值时再 `s_waitcnt vmcnt(0)`。
        # 同一个语句里连着发的几条 load 因此可以同时在飞（访存并行度）。
        v = Val("v", r, et, pending=True)
        self.pending_vals.append(v)
        return v

    def store(self, node: ast.Subscript, value: Val) -> None:
        if not isinstance(node.value, ast.Name):
            raise CompileError("只支持 base[index]")
        ptr = self.env.get(node.value.id)
        if ptr is None or ptr.kind != "ptr":
            raise CompileError(f"{node.value.id} 不是指针")
        idx = self.expr(node.slice)
        et = ptr.ty.split(":", 1)[1]
        if et == "f32":
            value = self._f32_operand(value)   # `y[i] = 1` 也要当 1.0
        src = self.as_vreg(value)
        lo, hi = self._addr(ptr, idx, et)
        mn = {"f32": "global_store_dword", "u32": "global_store_dword",
              "s32": "global_store_dword", "u16": "global_store_short",
              "s16": "global_store_short", "u8": "global_store_byte",
              "s8": "global_store_byte"}[et]
        self.emit(f"{mn} v[{lo}:{hi}], v{src}, off")
        self.emit("s_waitcnt vmcnt(0)")
        self.free_addr_pair()
        self._release(value)

    # ---------------- 语句 ----------------
    def stmt(self, node) -> None:
        self._stmt(node)
        self.tmp_v = self.tmp_base
        # 语句结束：所有临时值都死了（变量在专属寄存器区，从不落在临时池里），
        # 池子整体清空
        self.live_temps.clear()
        self.free_temps.clear()
        self.new_temps.clear()
        self.pending_vals.clear()

    def _stmt(self, node) -> None:
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
            op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
                  ast.Mod: "%", ast.BitAnd: "&", ast.BitOr: "|",
                  ast.BitXor: "^", ast.LShift: "<<", ast.RShift: ">>"}.get(type(node.op))
            if op is None:
                raise CompileError(f"增强赋值不支持 {type(node.op).__name__}")
            self._assign_name(t.id, self.binop(op, cur, rhs))
        elif isinstance(node, ast.If):
            self._if(node)
        elif isinstance(node, ast.For):
            self._for(node)
        elif isinstance(node, ast.While):
            self._while(node)
        elif isinstance(node, ast.Break):
            if self.varying_regions:
                raise CompileError(
                    "break 不能放在 varying 条件的 if 里：分支是标量指令、不受 "
                    "exec 掩码控制，会把**所有** lane 一起跳出循环（实测会让本来"
                    "不该退出的 lane 也退出）。请把退出条件并进循环条件，或把 "
                    "break 挪到 uniform 条件下面。")
            self.emit(f"s_branch {self.loop_stack[-1][1]}")
        elif isinstance(node, ast.Continue):
            if self.varying_regions:
                raise CompileError(
                    "continue 不能放在 varying 条件的 if 里：标量分支不受 exec "
                    "掩码控制，会把**所有** lane 一起跳回循环头。请把条件并进"
                    "循环条件，或把 continue 挪到 uniform 条件下面。")
            self.emit(f"s_branch {self.loop_stack[-1][0]}")
        elif isinstance(node, ast.Pass):
            return
        else:
            raise CompileError(f"不支持的语句 {type(node).__name__}")

    def _assign_name(self, name: str, val: Val, force_ty: str | None = None) -> None:
        """给局部变量赋值。

        标量（SGPR）变量是**整波一份**的：SGPR 指令不受 exec 掩码影响。所以
        一个标量变量只有在「它自己那个 varying 区」里做记账才是自洽的整波
        语义；一旦跑到**别的**区（或区外）去赋值，用户想要的几乎一定是
        per-lane 语义，这时整波记账就是静默算错——直接报错，让用户改用 f32
        局部变量（f32 天生是 per-lane 的 VGPR）。
        """
        if force_ty == "f32":
            val = self._f32_operand(val)        # `v: f32 = 1` 要当 1.0
        ty = force_ty or val.ty
        if name in self.env:
            dst = self.env[name]
            if dst.kind == "v":
                self.emit(f"v_mov_b32_e32 v{dst.reg}, v{self.as_vreg(val)}")
            elif dst.kind == "s":
                if self.is_uniform(val):
                    if self._region_of(name) != self._cur_region():
                        # SGPR 指令是**整波执行一次**的，不受 exec 掩码影响。
                        # 变量在 A 区建、却在另一个区（含区外）被赋值，说明
                        # 想要的是 per-lane 语义——这时整波记账会静默算错
                        # （实测 `c=0` 在区外、`if x[i]<0: c=c+1` 在区内 →
                        # 条件不成立的 lane 也一起变成 1，128 个元素里错 32 个）。
                        raise CompileError(
                            f"{name} 是 uniform（标量）变量，不能在**别的** varying "
                            f"区里赋值：SGPR 指令整波执行一次、不受 exec 掩码"
                            f"影响，所有 lane 都会看到这次修改——要的是 per-lane "
                            f"计数/累加时会**静默算错**。要做 per-lane 状态请用 "
                            f"f32 变量（`{name} = 0.0` 那样，f32 局部变量天生是 "
                            f"per-lane 的 VGPR）；要保留整波一致的标量记账，请把"
                            f"它和赋值放在同一个 varying 区里（或者挪到 uniform "
                            f"上下文）。")
                    self.emit(f"s_mov_b32 s{dst.reg}, s{self.as_sreg(val)}")
                else:
                    # 不把 varying 收回 uniform
                    r = self._named_vreg(name)
                    self.emit(f"v_mov_b32_e32 v{r}, v{self.as_vreg(val)}")
                    self.env[name] = Val("v", r, ty)
                    self.var_region[name] = self._cur_region()
            else:
                raise CompileError("不能赋值给指针")
            self._release(val)
            return
        # 新建变量：初值是 uniform 的整数/整波标量就建成 SGPR（区里建也一样，
        # 这样 dequant 那种「区里建、区里改」的地址/系数记账照旧是整波语义、
        # 产物不变）；f32 与 varying 初值才给 VGPR。_region_of 记下它是在哪个
        # varying 区里建的，跨区赋值时上面那条报错会挡住静默算错。
        if self.is_uniform(val) and ty != "f32":
            r = self._alloc_var_s()
            self.emit(f"s_mov_b32 s{r}, s{self.as_sreg(val)}")
            self.env[name] = Val("s", r, ty)
        else:
            r = self._named_vreg(name)
            self.emit(f"v_mov_b32_e32 v{r}, v{self.as_vreg(val)}")
            self.env[name] = Val("v", r, ty)
        self.var_region[name] = self._cur_region()
        self._release(val)

    def _if(self, node: ast.If) -> None:
        cond = self.expr(node.test)
        if cond.kind == "vpred":
            save = self.save_s
            self.save_s += 2
            end = self.new_label("if_end")
            self.region_n += 1
            self.varying_regions.append(self.region_n)
            if not node.orelse:
                self.emit(f"s_and_saveexec_b64 s[{save}:{save + 1}], vcc")
                self.emit(f"s_cbranch_execz {end}")
                for st in node.body:
                    self.stmt(st)
                self.emit(f"s_or_b64 exec, exec, s[{save}:{save + 1}]")
                self.label(end)
                self.varying_regions.pop()
                self.save_s -= 2
                return
            # varying if/else：把 exec 分别掩成 (old & cond) 与 (old & ~cond)，
            # 两条路径**都要跑**（同一个 wave 里两类 lane 都可能存在），所以不能
            # 用 s_branch 跳过 else。条件本身要另存一对 SGPR：then 里的地址计算
            # 会用 vcc 做进位输出，直接依赖 vcc 会在 else 之前被冲掉。
            cond_s = self._alloc_tmp_s(2)
            els = self.new_label("if_else")
            self.emit(f"s_mov_b64 s[{cond_s}:{cond_s + 1}], vcc")
            self.emit(f"s_and_saveexec_b64 s[{save}:{save + 1}], vcc")
            self.emit(f"s_cbranch_execz {els}")
            for st in node.body:
                self.stmt(st)
            self.label(els)
            self.emit(f"s_mov_b64 exec, s[{save}:{save + 1}]")
            self.emit(f"s_andn2_b64 exec, exec, s[{cond_s}:{cond_s + 1}]")
            for st in node.orelse:
                self.stmt(st)
            self.emit(f"s_mov_b64 exec, s[{save}:{save + 1}]")
            self.label(end)
            self.varying_regions.pop()
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

    def _while(self, node: ast.While) -> None:
        """`while cond:` —— 条件必须是 **uniform**（标量）比较。

        varying 条件需要 loop-carried 的 exec 掩码（每一轮都可能退出不同的
        lane），那套约定还没定，所以直接报错而不是给错的结果。
        `while 1:` 支持（无条件的无限循环，靠 `break` 退出）。

        循环变量要建在 uniform 上下文里：varying 区里建的标量是「整波记账」
        语义（跨区赋值会直接报错），拿它当条件就不再是 uniform 比较了。
        """
        head = self.new_label("while_head")
        end = self.new_label("while_end")
        self.label(head)
        cond = self.expr(node.test)
        unconditional = cond.kind == "lit" and cond.value not in (0, 0.0, None)
        if not unconditional:
            if cond.kind != "spred":
                raise CompileError("while 的条件必须是 uniform（标量）比较；"
                                   "varying 条件请用 for + break")
            self.emit(f"s_cbranch_scc0 {end}")
        self.loop_stack.append((head, end))
        for st in node.body:
            self.stmt(st)
        self.loop_stack.pop()
        self.emit(f"s_branch {head}")
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
        import json as _json
        (out_dir / f"{fn.name}.catalog.json").write_text(_json.dumps(
            {"version": 1, "kernels": [{
                "name": fn.name, "lookup": fn.name, "args": args,
                "kernarg_size": ksize, "group_segment": 0, "private_segment": 0}]},
            ensure_ascii=False, indent=1), encoding="utf-8")
        outputs.append(h_path)
    return outputs


def compile_file(path: pathlib.Path, out_dir: pathlib.Path, only: str | None = None):
    return compile_source(path.read_text(encoding="utf-8"), out_dir, only)
