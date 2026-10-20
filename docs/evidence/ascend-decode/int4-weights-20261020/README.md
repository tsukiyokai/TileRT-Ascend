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
