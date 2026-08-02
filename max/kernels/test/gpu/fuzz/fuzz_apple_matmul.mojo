# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
#
# Fuzz target: Apple (Metal) dense GEMM -- both kernel paths.
#
# Covers the two dense GEMMs Apple ships, selected by the `path` fuzz axis:
# `enqueue_apple_matmul` (M5 hardware MMA) and `gemm_kernel_apple_8x8` (the
# path the dispatcher uses on M1-M4, which also runs on M5). Fuzzing only the
# former would leave the target inert on every Apple GPU except an M5.
#
# This is the first Metal/Apple-GPU target in the fuzz suite. It differs from
# the NVIDIA targets in two ways that shape the design:
#
#   1. NO COMPTIME SHAPES. The tuned SM100 kernel reads N/K from the tensors'
#      STATIC shape, so `fuzz_matmul.mojo` has to pin them with `-D N=.. -D K=..`
#      and can only fuzz M. `AppleM5MatMul.run` derives M/N/K from the runtime
#      TileTensor dims, so M, N AND K are all runtime fuzz axes here -- the
#      ragged-tile space (`m % BM`, `n % BN`, `k % BK`) is reachable in one
#      binary. BM=BN=64 and BK=16 (32 on the NT fp16/bf16 `use_x2` path) are the
#      interesting moduli fed to `boundary_int`.
#
#   2. NO COMPUTE SANITIZER. `memcheck`/`initcheck`/`racecheck` are NVIDIA-only
#      and the redzone/poison device allocators are unvalidated on Metal, so the
#      memory-safety oracles that the NVIDIA targets default to are unavailable.
#      The default oracle is therefore `ref` (numerical correctness vs an
#      fp32-accum naive Metal kernel), backed by `determinism` and `contract`.
#
# Fuzz axes (spec fields; each is also a `--<key>` flag):
#   m, n, k   -- boundary-aware dims (ragged/aligned tiles, K tails).
#   tb        -- transpose_b: 0 = NN (B is [K, N]), 1 = NT (B is [N, K]). NT
#                fp16/bf16 selects the `use_x2` BK=32 double-strip MMA, a
#                different K-loop from NN's `k_unroll=4` path, so this flips the
#                kernel body and not just an index.
#   splitk    -- force_split_k: 0 = auto-route, 1 = always split-K, 2 = never.
#                The auto route only fires on under-occupied shapes, so 1 is the
#                only way to fuzz the split-K partial+reduce pair on the balanced
#                shapes `boundary_int` mostly draws.
#   dist      -- input value distribution (uniform/normal/sparse/all_equal).
#
# Oracles:
#   `ref` (default, --check 1)  -- vs a naive fp32-accum Metal reference kernel.
#   `determinism` (--rerun N)   -- re-launch the SAME input N times, require
#                bit-exact output. The single-pass kernel gives each output tile
#                one owning threadgroup and split-K reduces through a separate
#                deterministic reduce kernel, so there is no cross-threadgroup
#                atomic to reorder: any run-to-run difference is a real race.
#   `contract` (--contract 1)   -- NaN-propagation contract. Inject ONE NaN at
#                A[pm, pk] over otherwise-uniform inputs. Every C[pm, j] is a
#                dot product that reads that element, so it must be non-finite
#                for ALL j; every C[i, j] with i != pm must stay finite. This
#                catches both a dropped/skipped K element (row pm goes finite)
#                and a cross-tile write bleed (a clean row goes non-finite) --
#                the OOB class the missing Metal memcheck would otherwise cover.

from std.collections import Optional
from std.math import ceildiv, isfinite
from std.random import random_ui64, seed
from std.sys.defines import get_defined_dtype, get_defined_int
from std.utils.numerics import nan

from std.gpu import WARP_SIZE, global_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from layout import TileTensor
from layout.tile_layout import row_major
from linalg.matmul.gpu.apple import gemm_kernel_apple_8x8
from linalg.matmul.gpu.apple.matmul_kernel import enqueue_apple_matmul

from _fuzz import (
    VD_ALL_EQUAL,
    boundary_int,
    collect_args,
    fill_by_dist,
    fill_uniform,
    flag,
    flag_int,
    numeric_check,
    value_dist_name,
)

comptime in_dtype = get_defined_dtype["in_dtype", DType.float16]()
comptime c_dtype = get_defined_dtype["c_dtype", DType.float32]()

# Kernel path (a fuzz axis). Apple ships two dense GEMMs and the dispatcher picks
# between them by hardware, so fuzzing only one leaves the other unexercised:
#   PATH_M5   -- `AppleM5MatMul` via `enqueue_apple_matmul`: Metal 4 hardware MMA,
#                Apple M5 ONLY (raises on compute_capability != 5).
#   PATH_8X8  -- `gemm_kernel_apple_8x8`: the 8x8 simdgroup-matrix path the
#                dispatcher uses for M1-M4. Runs on ANY Apple GPU, including M5.
# The path is drawn per case rather than chosen from the live hardware so that a
# spec means the same thing everywhere and a corpus entry stays portable; a
# PATH_M5 case on non-M5 silicon reports FUZZ_SKIP instead of a bogus failure.
# The draw is deliberately skewed 1:3 toward PATH_8X8: M5 is rare while every
# other Apple GPU runs the 8x8 path, so an even split would burn half the budget
# on instant skips for nearly every machine this target actually runs on.
comptime PATH_M5 = 1
comptime PATH_8X8 = 2

# `gemm_kernel_apple_8x8` tiling, matching what the M1-M4 dispatcher enqueues.
comptime B8_BM = 64
comptime B8_BN = 64
comptime B8_NSG = 4

# `AppleM5MatMul` defaults: 64x64 threadgroup block, BK=16 K-strip. These are the
# moduli where a ragged tail is handled by a different code path than a full tile.
comptime TILE_MN = 64
comptime TILE_K = 16

# Bounded so the O(M*N*K) naive reference stays roughly sub-second per case: the
# worst-case 256*256*2048 is ~134M FMAs for the naive Metal kernel. 256 still
# spans 4 BM/BN tiles per dim, so multi-tile and ragged-edge coverage is intact.
comptime M_MAX = 256
comptime N_MAX = 256
comptime K_MAX = 2048

# Split-K is wired for the bf16/fp16 family only (`run_split_k_partial`), so the
# `splitk` axis collapses to auto-only for fp32 inputs.
comptime SPLIT_K_SUPPORTED = (
    in_dtype == DType.float16 or in_dtype == DType.bfloat16
)

comptime fuzz_seed = get_defined_int["fuzz_seed", 12345]()
comptime budget = get_defined_int["budget", 16]()


def _tolerances() -> Tuple[Float64, Float64]:
    """Returns (atol, rtol) for the `ref` oracle at `in_dtype`.

    Both the kernel and the reference read the SAME rounded inputs and both
    accumulate in fp32, so the expected gap is accumulation ORDER only. fp32
    inputs are the exception: the M5 MMA truncates fp32 operands to fp19, which
    the fp32 reference does not, so that path needs a much looser band.
    """
    comptime if in_dtype == DType.float32:
        return (Float64(0.5), Float64(3e-2))
    elif in_dtype == DType.bfloat16:
        return (Float64(0.2), Float64(2e-2))
    else:
        return (Float64(5e-2), Float64(1e-2))


# ===----------------------------------------------------------------------=== #
# Naive fp32-accum reference (a Metal kernel; the host equivalent is too slow at
# M*N*K = 256*256*2048).
# ===----------------------------------------------------------------------=== #


def naive_matmul_ref_kernel[
    transpose_b: Bool
](
    c: UnsafePointer[Scalar[c_dtype], MutAnyOrigin],
    a: UnsafePointer[Scalar[in_dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[in_dtype], MutAnyOrigin],
    m_dev: Int32,
    n_dev: Int32,
    k_dev: Int32,
):
    """C[i,j] = sum_k A[i,k]*B[k,j] (NN) or sum_k A[i,k]*B[j,k] (NT), fp32 accum.

    One thread per output element in the plainest possible order. It shares no
    tiling, no simdgroup MMA and no loader with `AppleM5MatMul`, so a tiling or
    ragged-edge bug in the kernel under test cannot be mirrored here.
    """
    # `Int` is not device-passable; widen the fixed-width args.
    var m = Int(m_dev)
    var n = Int(n_dev)
    var k = Int(k_dev)
    var col = global_idx.x
    var row = global_idx.y
    if row < m and col < n:
        var acc = Float32(0)
        for k_i in range(k):
            var av = a[row * k + k_i].cast[DType.float32]()
            comptime if transpose_b:
                acc += av * b[col * k + k_i].cast[DType.float32]()  # B is [N,K]
            else:
                acc += av * b[k_i * n + col].cast[DType.float32]()  # B is [K,N]
        c[row * n + col] = acc.cast[c_dtype]()


# ===----------------------------------------------------------------------=== #
# Case spec
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct CaseSpec(Copyable, Movable, Writable):
    var m: Int
    var n: Int
    var k: Int
    var tb: Int  # 0 = NN, 1 = NT (transpose_b)
    var splitk: Int  # 0 = auto, 1 = force on, 2 = force off (PATH_M5 only)
    var dist: Int  # value-distribution id (see _fuzz.VD_*)
    var path: Int  # kernel path: PATH_M5 or PATH_8X8

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "m=",
            self.m,
            " n=",
            self.n,
            " k=",
            self.k,
            " tb=",
            self.tb,
            " splitk=",
            self.splitk,
            " dist=",
            self.dist,
            " (",
            value_dist_name(self.dist),
            ") path=",
            self.path,
            " (",
            "m5" if self.path == PATH_M5 else "8x8",
            ")",
        )


def _auto_mix_dist() -> Int:
    """Draws a value distribution for the `ref` auto-mix.

    Excludes both `specials` (NaN/Inf) and `large`: at max_finite operands every
    dot product overflows to Inf by construction, in the kernel and the reference
    alike, so such a case can only ever compare Inf against Inf -- it burns
    budget without being able to fail. Both stay reachable via an explicit
    `--dist`, where `large` is a saturation probe and `specials` feeds the
    `contract` oracle.
    """
    var d = Int(random_ui64(0, 3))
    return VD_ALL_EQUAL if d == 3 else d  # {uniform, normal, sparse, all_equal}


def gen_specs(n: Int) -> List[CaseSpec]:
    var specs = List[CaseSpec]()
    for _ in range(n):
        specs.append(
            CaseSpec(
                boundary_int(1, M_MAX, TILE_MN),
                boundary_int(1, N_MAX, TILE_MN),
                boundary_int(1, K_MAX, TILE_K),
                Int(random_ui64(0, 1)),
                Int(random_ui64(0, 2)) if SPLIT_K_SUPPORTED else 0,
                _auto_mix_dist(),
                PATH_M5 if random_ui64(0, 3) == 0 else PATH_8X8,
            )
        )
    return specs^


def _force_split_k(splitk: Int) -> Optional[Bool]:
    """Maps the `splitk` spec field to `enqueue_apple_matmul`'s override."""
    if splitk == 1:
        return Optional[Bool](True)
    if splitk == 2:
        return Optional[Bool](False)
    return Optional[Bool](None)


def _launch[
    transpose_b: Bool
](
    ctx: DeviceContext,
    path: Int,
    splitk: Int,
    mut c_dev: DeviceBuffer[c_dtype],
    a_dev: DeviceBuffer[in_dtype],
    b_dev: DeviceBuffer[in_dtype],
    m: Int,
    n: Int,
    k: Int,
) raises:
    """Wraps the device buffers as TileTensors and launches the selected path.

    Takes raw buffers rather than tensors so the rerun (`determinism`) loop can
    re-launch through the exact same code path as the first launch.
    """
    var b_rows = n if transpose_b else k
    var b_cols = k if transpose_b else n
    var a_tt = TileTensor(a_dev.unsafe_ptr(), row_major(m, k)).as_immut()
    var b_tt = TileTensor(
        b_dev.unsafe_ptr(), row_major(b_rows, b_cols)
    ).as_immut()
    var c_tt = TileTensor(c_dev.unsafe_ptr(), row_major(m, n))

    if path == PATH_M5:
        enqueue_apple_matmul[
            in_type=in_dtype, c_type=c_dtype, transpose_b=transpose_b
        ](c_tt, a_tt, b_tt, ctx, _force_split_k(splitk))
    else:
        comptime kernel = gemm_kernel_apple_8x8[
            c_dtype,
            in_dtype,
            in_dtype,
            type_of(c_tt).LayoutType,
            type_of(a_tt).LayoutType,
            type_of(b_tt).LayoutType,
            type_of(c_tt).Storage,
            type_of(a_tt).Storage,
            type_of(b_tt).Storage,
            transpose_b,
            BLOCK_M=B8_BM,
            BLOCK_N=B8_BN,
            NUM_SIMDGROUPS=B8_NSG,
        ]
        ctx.enqueue_function[kernel](
            c_tt,
            a_tt,
            b_tt,
            Int32(m),
            Int32(n),
            Int32(k),
            grid_dim=(ceildiv(n, B8_BN), ceildiv(m, B8_BM)),
            block_dim=(B8_NSG * WARP_SIZE,),
        )


# ===----------------------------------------------------------------------=== #
# Case execution
# ===----------------------------------------------------------------------=== #


def _run_case[
    transpose_b: Bool
](
    ctx: DeviceContext,
    spec: CaseSpec,
    check: Bool = False,
    rerun: Int = 0,
    contract: Bool = False,
) raises:
    var m = spec.m
    var n = spec.n
    # The 8x8 dispatch gate takes K as a multiple of 16, so round a fuzzed ragged
    # K up to the next multiple for that path. The M5 path takes K exactly as
    # drawn, so ragged-K tails stay covered there. Rounding (rather than
    # rejecting the case) keeps every drawn shape useful on both paths, and it is
    # deterministic, so a corpus repro replays identically.
    var k = spec.k if spec.path == PATH_M5 else ceildiv(spec.k, 16) * 16
    var a_size = m * k
    var b_size = n * k  # [K, N] or [N, K]: same element count either way.
    var c_size = m * n

    var a_host = ctx.enqueue_create_host_buffer[in_dtype](a_size)
    var b_host = ctx.enqueue_create_host_buffer[in_dtype](b_size)

    # The NaN probe's row/col are derived from the spec so they move with the
    # shape (and so a shrunk repro reproduces the same probe).
    var probe_m = (spec.k * 7 + 3) % m
    var probe_k = (spec.m * 11 + 5) % k

    if contract:
        fill_uniform(a_host.as_span())
        fill_uniform(b_host.as_span())
        a_host[probe_m * k + probe_k] = nan[in_dtype]()
    else:
        fill_by_dist(a_host.as_span(), spec.dist)
        fill_by_dist(b_host.as_span(), spec.dist)

    var a_dev = ctx.enqueue_create_buffer[in_dtype](a_size)
    var b_dev = ctx.enqueue_create_buffer[in_dtype](b_size)
    var c_dev = ctx.enqueue_create_buffer[c_dtype](c_size)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    _launch[transpose_b](
        ctx, spec.path, spec.splitk, c_dev, a_dev, b_dev, m, n, k
    )
    ctx.synchronize()

    if rerun > 0:
        # Run-to-run bit stability. Each output tile has exactly one owning
        # threadgroup (single-pass) or is reduced by a separate deterministic
        # reduce kernel (split-K), so there is no cross-threadgroup accumulation
        # whose order could legitimately vary: any diff is a real race.
        var first_h = ctx.enqueue_create_host_buffer[c_dtype](c_size)
        ctx.enqueue_copy(first_h, c_dev)
        ctx.synchronize()
        for _ in range(rerun - 1):
            _launch[transpose_b](
                ctx, spec.path, spec.splitk, c_dev, a_dev, b_dev, m, n, k
            )
            ctx.synchronize()
            var rep_h = ctx.enqueue_create_host_buffer[c_dtype](c_size)
            ctx.enqueue_copy(rep_h, c_dev)
            ctx.synchronize()
            if not numeric_check(
                rep_h.as_span(), first_h.as_span(), atol=0.0, rtol=0.0
            ):
                # DRIV-199: keep device buffers alive past `synchronize`.
                _ = a_dev^
                _ = b_dev^
                _ = c_dev^
                raise Error("apple matmul run-to-run nondeterminism (rerun)")
    elif contract:
        var c_h = ctx.enqueue_create_host_buffer[c_dtype](c_size)
        ctx.enqueue_copy(c_h, c_dev)
        ctx.synchronize()
        var n_probe_finite = 0  # probe row elements that WRONGLY stayed finite
        var n_clean_nonfinite = (
            0  # clean row elements that WRONGLY went NaN/Inf
        )
        for i in range(m):
            for j in range(n):
                var fin = isfinite(c_h[i * n + j].cast[DType.float64]())
                if i == probe_m:
                    if fin:
                        n_probe_finite += 1
                elif not fin:
                    n_clean_nonfinite += 1
        if n_probe_finite > 0 or n_clean_nonfinite > 0:
            print(
                "FUZZ_CONTRACT_FAIL NaN-propagation probe_m=",
                probe_m,
                "probe_k=",
                probe_k,
                "probe_row_finite=",
                n_probe_finite,
                "/",
                n,
                "clean_rows_nonfinite=",
                n_clean_nonfinite,
                "/",
                (m - 1) * n,
            )
            _ = a_dev^
            _ = b_dev^
            _ = c_dev^
            raise Error("apple matmul NaN-propagation contract violated")
    elif check:
        var c_ref_dev = ctx.enqueue_create_buffer[c_dtype](c_size)
        comptime BX = 16
        comptime BY = 16
        ctx.enqueue_function[naive_matmul_ref_kernel[transpose_b]](
            c_ref_dev.unsafe_ptr(),
            a_dev.unsafe_ptr(),
            b_dev.unsafe_ptr(),
            Int32(m),
            Int32(n),
            Int32(k),
            grid_dim=(ceildiv(n, BX), ceildiv(m, BY)),
            block_dim=(BX, BY),
        )
        ctx.synchronize()
        var c_h = ctx.enqueue_create_host_buffer[c_dtype](c_size)
        var c_ref_h = ctx.enqueue_create_host_buffer[c_dtype](c_size)
        ctx.enqueue_copy(c_h, c_dev)
        ctx.enqueue_copy(c_ref_h, c_ref_dev)
        ctx.synchronize()
        var tol = _tolerances()
        var ok = numeric_check(
            c_h.as_span(), c_ref_h.as_span(), atol=tol[0], rtol=tol[1]
        )
        _ = c_ref_dev^
        if not ok:
            _ = a_dev^
            _ = b_dev^
            _ = c_dev^
            raise Error("apple matmul vs fp32-accum naive mismatch")

    # DRIV-199 workaround: keep device buffers alive past `synchronize`, else
    # ASAP destruction frees them mid-kernel and the run flakes.
    _ = a_dev^
    _ = b_dev^
    _ = c_dev^


def run_one_case(
    ctx: DeviceContext,
    spec: CaseSpec,
    cc: Int,
    check: Bool = False,
    rerun: Int = 0,
    contract: Bool = False,
) raises:
    """Dispatches on the runtime `tb` axis to the two comptime instantiations.

    A PATH_M5 case on non-M5 silicon is skipped rather than run: only that path
    needs M5, so skipping per case (instead of aborting the whole run) keeps the
    PATH_8X8 cases executing on M1-M4.
    """
    if spec.path == PATH_M5 and cc != 5:
        print(
            "FUZZ_SKIP reason=path-m5-requires-apple-m5 compute_capability=", cc
        )
        return
    if spec.tb == 1:
        _run_case[True](ctx, spec, check, rerun, contract)
    else:
        _run_case[False](ctx, spec, check, rerun, contract)


def main() raises:
    var args = collect_args()
    var mode = flag(args, "--mode", "fuzz")
    var the_seed = flag_int(args, "--seed", fuzz_seed)
    var the_budget = flag_int(args, "--budget", budget)
    var check = flag_int(args, "--check", 0) == 1
    var rerun = flag_int(args, "--rerun", 0)
    var contract = flag_int(args, "--contract", 0) == 1
    seed(the_seed)

    if mode == "list-specs":
        var specs = gen_specs(the_budget)
        for i in range(len(specs)):
            print(
                "FUZZ_SPEC idx=",
                i,
                "m=",
                specs[i].m,
                "n=",
                specs[i].n,
                "k=",
                specs[i].k,
                "tb=",
                specs[i].tb,
                "splitk=",
                specs[i].splitk,
                "dist=",
                specs[i].dist,
                "path=",
                specs[i].path,
            )
        return

    with DeviceContext() as ctx:
        var cc = ctx.compute_capability()
        if mode == "single":
            var spec = CaseSpec(
                flag_int(args, "--m", 128),
                flag_int(args, "--n", 128),
                flag_int(args, "--k", 128),
                flag_int(args, "--tb", 0),
                flag_int(args, "--splitk", 0),
                flag_int(args, "--dist", 0),
                flag_int(args, "--path", PATH_8X8),
            )
            print("FUZZ_SINGLE ", spec)
            run_one_case(ctx, spec, cc, check, rerun, contract)
            print("FUZZ_RESULT verdict=PASS")
            return

        print(
            "=== fuzz_apple_matmul seed=",
            the_seed,
            "budget=",
            the_budget,
            "in_dtype=",
            in_dtype,
            "c_dtype=",
            c_dtype,
            "compute_capability=",
            cc,
            "===",
        )
        var specs = gen_specs(the_budget)
        for i in range(len(specs)):
            print("case", i, ":", specs[i])
            run_one_case(ctx, specs[i], cc, check, rerun, contract)
        print("=== done:", len(specs), "cases ===")
