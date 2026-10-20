# The INT4 weights the chip can read and the library will not pass

The routed-expert kernel streams INT8 weights and is
[weight-streaming bound at 69% of peak](../routed-kernel-20261020/README.md).
The model is W4A8. So the dominant term is twice the size it has to be.

## The vendor does read INT4, and the profile says so

From this project's own capture, `kernel_details.csv` on rank zero:

| op | input data types | input formats |
|---|---|---|
| `GroupedMatmulSwigluQuantV2` | `INT8;FLOAT;INT64;DT_INT4;UINT64;FLOAT` | `ND;ND;ND;FRACTAL_NZ;ND;ND` |
| `GroupedMatmul` | `INT8;DT_INT4;FLOAT;UINT64;FLOAT;FLOAT16;FLOAT16;INT64;FLOAT` | `ND;FRACTAL_NZ;ND;NCL;ND;ND;ND;ND;ND` |

The weight is `DT_INT4` in `FRACTAL_NZ`, and the op's own name says it:
`aclnnGroupedMatmulSwigluQuantWeightNzV2`. So A3 reads 4-bit weights.

That reframes the comparison. Per rank per layer the routed experts are
sixteen experts of 2,359,296 elements: 18.9 MB at INT4, 37.7 MB at INT8.
The vendor's 165.2 µs moves 18.9 MB, which is 114 GB/s -- 10% of the die's
1.155 TB/s. This project's kernel moves 37.7 MB in 49.3 µs, which is
765 GB/s, 66%. So it is 3.3 times faster while carrying twice the bytes, and
its bandwidth efficiency is 6.7 times better. The vendor op is not
bandwidth-bound by any reading.

At 66% efficiency on 18.9 MB the kernel would take 24.8 µs a layer. That is
the prize: half of the biggest lever this project has.

## Where the high-level library stops

`MatmulType<TPosition::GM, CubeFormat::*, int4b_t, *>` fails to compile in
every combination tried -- NZ and ND, transposed and not, under `CFG_NORM`,
`CFG_MDL` and `CFG_IBSHARE_NORM`. Seven variants, two failure sites:
`data_copy_wrapper_utils.h:312` for NZ and `data_copy_wrapper_nd.h:111` for
ND, both `no matching function for call to 'DataCopy'`.

The instantiation names the reason. The failing call is
`CopyNZ2NZImpl<integer_sub_type<4, true>, signed char, false>`, and every
frame carries `MatmulInputBType<MatmulType<...int4b_t...>, signed char>`: the
library's plan is that global memory holds INT4 while the L1 tile holds INT8,
so the copy would have to widen on the way in. A3's MTE2 does not widen, so
the overload does not exist. The surrounding code expects it to be possible
-- `if constexpr (IsSupportB4<TransT>()) { blockLen /= INT4_TWO; ... }` is
right there in the NZ path -- which makes this a gap in the abstraction
rather than a limit of the part.

## The chip's own path is open

Six compile probes, all accepted on `--npu-arch=dav-2201`:

| what | verdict |
|---|---|
| `DataCopy(LocalTensor<int4b_t> in UB, GlobalTensor<int4b_t>, count)` | accepted |
| the same with `DataCopyParams` -- the call the library failed on | accepted |
| the same bytes as INT8, half the count | accepted |
| `DataCopy` into `TBuf<TPosition::B1>` as INT4 | accepted |
| `DataCopy` into `TBuf<TPosition::A1>` as INT4 | accepted |
| `DataCopy` into `TBuf<TPosition::A2>` as INT4 | accepted |

So INT4 reaches L1 and L0 as INT4. The other half of the journey is already
written in the library this project cannot use: `load_to_l0_utils.h` returns
`LoadL0bInstrType::LOAD2DTRANSPOSE` for `int4b_t`, and
`load_to_l0b_loadInstr.h` has an `IsSameType<TransT, int4b_t>` branch.

What is blocked is the high-level `Matmul`, not the hardware. Reaching the
24.8 µs means managing L1 and L0 by hand -- `DataCopy` the INT4 tile to L1,
`LoadData` it to L0B, `Mmad` -- instead of `IterateAll`.

## How the part does W4A8 at all: two INT4 mads a weight tile

Driving L1 and L0 by hand got one error, and it was the useful kind.
`kernel_operator_mm_impl.h:391` wants `__cb__ signed char *` for the third
operand, and the assertion beside it lists what `Mmad` takes:

> dst: int32_t, fm: int8_t, filter: int8_t; dst: float, fm: half, filter:
> half; dst: float, fm: float, filter: float; dst: float, fm: bfloat16_t,
> filter: bfloat16_t; dst: int32_t, fm: int4b_t, filter: int4b_t

So INT4 is there, but only with *both* operands INT4, dispatching to
`mad_s4`. The intrinsics on this part are `mad`, `mad_s4`, `mad_sp` and
`mad_mx` -- nothing mixed. An INT8 activation against an INT4 weight is not
an instruction here.

And a vector expansion of the weights is not an answer either: 37.7 M weight
elements a rank-layer would have to pass through the vector units against a
49.3 µs budget, which is orders out.

The vendor's own source says what it does instead. In
`opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/grouped_matmul_swiglu_quant/op_kernel/`
there are three files named `grouped_matmul_swiglu_quant_a8w4_msd_pre.h`,
`_mid.h` and `_post.h`. The pre stage holds `xHighI4Tensor` and
`xLowI4Tensor` and masks with `Duplicate(xLowI16Tensor, 0x0F0F)`, commented
"get rid of high 4 bits in every int8"; its buffer accounting reads
`M * K * 7B (1B + 0.5B + 0.5B + 2B + 4B)`.

So the INT8 activation is split into two INT4 halves, each is multiplied by
the INT4 weight, and the products are recombined as 16 × high + low. Two
`mad_s4` calls a weight tile.

That is a good trade here and a bad one for the vendor. The split is over the
activation, which at batch-one decode is 12 KB an expert against 2.36 MB of
weights; the weight tile in L0B serves both mads, so L1 traffic is unchanged
and HBM traffic halves; and the cube is weight-streaming bound at 66% of
peak, not mad bound, so doubling the mads costs little. The vendor pays for
it by splitting pre, mid and post into three kernels with global memory
between them -- which is a plain reading of why its 165.2 µs is 10% of
bandwidth. One kernel keeps the split in UB.

## The primitive, driven by hand and measured

`DataCopy` global to L1, `LoadData` L1 to L0, `Mmad`, `Fixpipe` L0C to global.
M, K and N are 16, 64 and 16, one INT4 fractal of K a side, and both operands
are filled so that the answer does not depend on the fractal order.

| right-hand filling | expected | got |
|---|---|---|
| `0x11`, every nibble a one | 64, which is K | 64 everywhere, nothing untouched |
| `0x21`, nibbles of one and two | 96 | 96 everywhere |

The second row is the one that settles it. If the part read one nibble a byte
the answer would be 64 or 128; 96 is 32 ones and 32 twos, so both halves of
every byte are separate INT4 elements. `mad_s4` consumes packed INT4
correctly.

One negative worth keeping: `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY)`
is rejected at submission, not at run -- the queue reports `sqe_type=3(place
holder), errType=0x20(sq sw status error), sqSwStatus=0x23` and the launch
returns 507015 with no aicore exception anywhere in the log. The `__mix__(1, 2)`
shape this project already runs cube code through works, with the vector half
returning immediately.

So every link is proven: INT4 from global memory to L1, L1 to L0, the mad
itself, and the product out through the fixpipe. What remains is tiling
K = 6144 and N = 256 by hand, and the activation split the vendor's `a8w4_msd`
stages describe.
