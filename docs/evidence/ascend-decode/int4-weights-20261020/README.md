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

## The way through: a homogeneous INT4 pair, stacked along M

Hand-driving L1 and L0 got the primitive right and then stalled on the
micro-synchronisation of a reused L0 slot. With `PipeBarrier<PIPE_ALL>` the
same code answered a K of 128 exactly, 256 as 192, 512 as noise, 1024 as half
and 6144 exactly -- a hazard's signature, not arithmetic's. Replacing it with
`SetFlag`/`WaitFlag` on `MTE1_M` and `M_MTE1` made K of 512 and above fail at
launch instead. Along the way the layout did come out: the INT4 fractal is
`[k-fractal][16 columns][64 of K]`, a prefix of 64 nibbles lands entirely in
column zero and a prefix of 1024 spreads 64 across all sixteen; one mad covers
at most 128 of K; and eight fractals in one `LoadData` lose a mad's worth.

The way through is not to hand-roll at all. The library rejects INT4 because
it plans to widen it to INT8 on the way into L1 -- so give it a pair that
needs no widening. `MatmulType<..., int4b_t, ...>` for *both* operands
compiles in all four combinations tried: B as ND and as NZ, transposed and
not.

Two INT4 matmuls would read the weight twice and give back the saving, so the
two halves of the activation stack along M instead. With
`A' = [A_high ; A_low]` of 2 × ROWS by 6144 and one `Matmul`, the weight is
read once and

    C[r][n] = 16 × C'[r][n] + C'[ROWS + r][n]

M doubling from two rows to four costs almost nothing in a weight-bound
matmul, and the library's tiling -- the thing already reaching 66% of
bandwidth on INT8 -- is kept rather than reimplemented.

So the routed kernel's INT4 form is: the vector half splits the INT8
activation into two INT4 halves and stacks them, the cube runs one INT4
matmul an expert, and the vector half recombines with the shift before the
dequantisation it already does. The down projection takes the same treatment,
its activation being the intermediate the vector has just quantised.

## The INT4 kernel runs, and one defect is left

`routed_int4` builds and runs the whole chain -- vector split, INT4 matmul,
recombine, activate, requantise, split again, INT4 matmul, recombine, BF16 --
in one launch. Four experts on four blocks take 39.350 µs against the INT8
form's 47.108, which is the halved weight traffic showing through. It is not
correct yet: 35530 of 49152 values deviate.

The defect is isolated to one thing. An INT4 `DataCopy` to global memory lands
when its destination offset is zero and lands nowhere when it is not:

- The high halves, written at offset zero, are exact -- 98 → 6, 112 → 7,
  −73 → −5, 80 → 5, 13 → 0, −106 → −7, 24 → 1, which is `floor(x / 16)` every
  time.
- The low halves, written at nibble 12288, appear nowhere in the allocation. A
  search of the whole buffer for the expected sequence returns −1, and the
  bytes that should hold them read the same eight values across completely
  different input data -- they are a previous run's residue.
- Swapping which call goes first changes nothing, so it is not the second
  write that fails.
- Writing the known-good high half to the suspect offset also fails, so it is
  the offset and not the source.
- Three addressings fail alike: a `GlobalTensor<int4b_t>` subscript, a
  `__gm__ int4b_t*` offset, and that offset taken through a byte pointer.

The same `DataCopy` at offset zero, with and without `DataCopyParams`, works
in a plain vector kernel -- six probes above say so -- so the next cut is a
non-zero destination offset in that isolated probe rather than in this kernel.

## What the sanitizer was for

Three defects came out of `mssanitizer --tool=memcheck` in one run, each as
"illegal write of size 32, 0 bytes after the nearest allocation", naming the
buffer by its size:

- 512 bytes: `middle`. 24576: `stacked`. 8192: `partial`. An expert's offset
  was being computed in nibbles and applied to a pointer, so every expert
  wrote two regions along.
- Then reads of 928 and writes of 992 bytes off the ends: the cube works in
  sixteen-row tiles and the `Matmul` library reads and writes the whole tile
  whatever M says, while M here is four. Every buffer along M now carries a
  tile.

Each fix halved the deviating count -- 44042, then 25458 -- which is how a
real defect behaves and how a guess does not.

## Two things the algorithm owes to being told not to be naive

The split was eight float operations computing `((x + 8) mod 16) - 8` and then
the remainder. It is now five, and it starts from `floor(x / 16)`, which is an
INT4 for every INT8 rather than only for those within 119. The eight that the
low half still owes comes back as `8 * sum(w)` a column -- a function of the
weight alone, so a loader computes it once and the epilogue adds it where it
is already scaling. No clamp, no correction term, no range lost.

The test data was an arithmetic progression, whose low nibbles are a sawtooth
of period sixteen. A search for where a half actually landed then matches at
every aligned phase and answers almost anywhere; worse, a wrong answer can
hide behind a right one. With a pseudo-random generator the same search
returns −1 and means it.
