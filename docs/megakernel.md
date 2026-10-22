# AscendC decode fusion

> **What fusing a layer is worth, measured 2026-10-22.** The whole layer in one
> launch saves 10 microseconds a layer against three native launches, and the
> native layer at 574 microseconds ties the vendor engine's 575. Fusing launches
> is not the lever, because the cost is a **pass over the data** and a pass
> costs the same inside a kernel as outside it: a Torch reduction over 98304
> floats takes 11.82 microseconds and an in-kernel AscendC pass over 73728 takes
> 12.46, with the increment surviving graph capture.
>
> What pays is several operations sharing **one load**. An extra operation folded
> into an existing load costs 2.35 microseconds; one that brings its own load
> costs 24.30 -- a factor of 10.3. The cube obeys the same rule: an extra matmul
> invocation's whole cost is one more walk of its reduction, 4.41 microseconds
> per thousand of K, and a walk is the weight's read. That is where the front
> stage's 26.87 microseconds comes from -- four concatenated projections walking
> K once instead of four times.
>
> A layer makes about 48 passes over its data and reads 116.9 MB a rank, so
> `layer = 12 us * passes + 72.7 us`. The 2.9 millisecond target allows about
> eleven passes. Six are accounted for. The evidence is under
> `evidence/ascend-decode/`: `cost-is-per-pass-20261022`,
> `shared-load-20261022`, `pass-budget-20261022`,
> `invocation-has-no-fixed-cost-20261022`.


The implementation provides native stages and a one-launch layer for GLM-5.2
W4A8C8 with TP16 shard dimensions, EP disabled, and 1–32 verification rows.
The local layer passes single-die comparisons; the entry containing both TP
reductions passes a two-die comparison. These use synthetic parameters.
A three-rank cancellation test exposes a difference between its serial FP32
sum and the installed deterministic HCCL tree. A private tree reducer now
matches HCCL on ordinary and special-value inputs; integration into the TP
layer remains required.
A whole-model result and 2 ms end-to-end TPOT remain acceptance targets.
The actual-decoder eager gate passes M1/M2/M12/M32 with eight changing inputs
each: all 816 boundaries match after the M1 and M12 key-projection corrections.
The complete native FFN also passes both actual decoder weight sets at those
four widths: all 448 arithmetic boundaries and 256 immediate graph-consumer
checks match in the corrected split/terminal composition with static GU tiling.
These gates use synthetic parameters and explicit local TP cuts.
Whole-Decoder graph replay passes all four widths and both residual entry
modes: 128 replays and 3,264 raw-byte checks match. This comparison uses native
attention and the complete native FFN against the original attention/FFN chain.
The original final-reduction helper and separate layer/stream workspace owners
are retained; both TP boundaries are explicit local cuts.
The one-launch local layer also passes the separate-component comparison:
two consecutive indexed/shared layers, M1/2/12/32, and 3,488 raw-byte checks.
That gate uses eager execution with the actual local attention output feeding
FFN and both TP reductions replaced by identities. Its official ACLGraphWrapper
comparison also passes all four widths: four captures, 32 changing-input
replays and 3,488 raw-byte checks, including the complete active expert tables.
The persistent-top-k variant also passes 3,488 eager byte checks at all four
widths. Its selector writes the retained indices directly, removing the copy
before the next shared-indexer layer; unused and inactive regions stay unchanged.
With the [accepted N512 FFN preparation](../tests/glm52/whole_layer/layer_down_n512/README.md),
this variant passes both eager and official ACL graph comparisons at all four
widths. Each gate checks 3,488 numerical boundaries and all 200 Down tiling
bytes 136 times; the graph gate includes four captures and 32 changing replays.
Its [matched M12 timing](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-layer-pair-timing-v1-a3-194/summary.json)
rejects direct Top-K output: two scratch arenas give paired median regressions
of 14.26 and 9.09 us per indexed/shared two-layer pair, with no wins in 60
pairs. Both arms use N512; the original layer with its Top-K copy is retained.
The workload repeats two synthetic weight sets with identity TP reductions.
With the original Top-K copy retained, the [N512 whole-layer timing](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-layer-native-timing-v3-a3-194/summary.json)
improves on three native norm/attention/FFN launches per layer. The two scratch
arenas give paired median savings of 20.22 and 21.21 us per indexed/shared
two-layer pair, with all 60 pairs favoring fusion. The timed reference writes
the same owners directly, without norm-output copies. All 5,232 intermediate
and 600 event-tail byte checks pass. This is a one-bank local measurement;
TP reductions and whole-model TPOT remain outside its scope.

The [layer containing both TP reductions](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-tp-chain-wide-v2-a3-194/summary.json)
passes two-rank eager execution on A3 at M1/2/12/32, with eight changing
inputs at each width. All 7,232 intermediate byte checks and 256 comparisons
with real HCCL SUM pass. Rank-local projection and expert weights differ;
the reduced outputs and retained residuals agree across ranks. One SHMEM
window persists across both layers and all widths without host epoch resets.
Its [official graph comparison](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-tp-graph-wide-v1-a3-194/summary.json)
also passes all four widths: eight captures and 64 changing-input replays
across two ranks, with another 7,232 intermediate byte checks and 256 HCCL
comparisons. All 904 observed boundaries remain unchanged during capture;
replay runs no Python. The window stays on the actual capture stream across
all widths. This entry uses the static-GU FFN parent.
Its [paired M12 graph timing](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-tp-timing-m12-v1-a3-194/summary.json)
compares the complete TP entry with separate native stages and the project's
FP32 HCCL reduction arithmetic. Two arenas save 23.049/23.264 us per
indexed/shared two-layer pair, with all 60 pairs faster. Candidate/control
medians are 1125.374/1148.304 and 1130.584/1153.384 us. Each graph repeats
the pair sixteen times; each event contains thirty replays. Every arm uses
the maximum of its two rank-local event intervals before paired subtraction.
The one-pair-per-graph sensitivity check saves 32.449/31.283 us, with 59 of
60 pairs faster. All 18,532 production, 640 FP32-conversion/reduction and
4,320 event-tail byte checks pass. This comparison changes both layer
orchestration and collective implementation; it uses two controlled
TP16-shaped shards with repeated synthetic weights. TP16 and whole-model
TPOT acceptance remain required.
The [three-rank order diagnostic](evidence/ascend-decode/megakernel-device-20261021/integration/communication-order-three-rank-v1-a3-194/AUDIT.md)
demonstrates a numerical difference at M1 and M12: BF16 partials
`[1, 2^-25, -1]` produce zero with the current serial reduction and `2^-25`
with the installed FP32 HCCL reduction. Across 36 eager/graph cases, the
native result matches its serial oracle and HCCL matches the selected tree.
Runtime logs identify `AllReduceMeshAivFor91093Executor`, FP32 SUM and
deterministic execution. The existing two-rank results remain valid; the
serial order is unsuitable as a general replacement for this HCCL path.
The [installed binary and source audit](evidence/ascend-decode/megakernel-device-20261021/integration/tp16-reduction-order-source-audit-v1/README.md)
identifies the deterministic tree: each round folds the largest power-of-two
offset strictly below the live rank count. Its TP16 distances are 8, 4, 2, 1.
The three-rank inputs distinguish the serial expression from that tree;
TP16 runtime selection and vector-versus-atomic arithmetic remain separate
acceptance checks. C2's all-to-all BF16 payload also grows differently from
the installed FP32 partitioned collective: its per-rank off-chip payload is
four times larger at TP16 for these divisible shapes. This is a byte count,
not a measured TP16 latency.
The [private tree-reducer device comparison](evidence/ascend-decode/megakernel-device-20261021/integration/communication-tree-three-rank-v1-a3-194/AUDIT.md)
passes all 72 three-rank cases at M1/M12, including 32 special-value columns.
The candidate and HCCL match byte for byte on cancellation, signed zeros,
BF16 subnormals, finite inputs whose sums overflow, and final BF16 ties.
All 144 immediate-consumer checks pass. Six selector records and 72 launch
records identify the same deterministic FP32 AIV path. The test executes
both reductions and a dependent CUBE stage, but compares only the first
reduction and its consumer; whole-layer integration and TP16 remain required.

The [checkpoint recovery](evidence/ascend-decode/megakernel-device-20261021/integration/checkpoint-restore-v1-a3-194/README.md)
finds 108 original files on the NAS and restores the missing
`optional/quarot.safetensors` from the pinned ModelScope revision. The actual
consumer container resolves all 109 files at their expected sizes; the config
and recovered-file hashes match. The 95 weight shards were not rehashed.
The NAS and host-owned supplement survive container deletion. Whole-model
runtime restoration and a new steady-state baseline remain required.

| Stage | Native scope | Integration |
| --- | --- | --- |
| [Routed experts](../tests/glm52/routed_int4/README.md) | INT4 MSD split, gate/up, SwiGLU/requantization, down, original-slot combine | `--routed-kernel` replaces the routed body after top-8 selection |
| [MLA tail](../tests/glm52/mla_tail/README.md) | Value-up, BF16 boundary, dynamic A8 quantization, local output projection | `--mla-tail` replaces the continuous projection tail after SFA |
| [Indexed attention](../tests/glm52/whole_layer/attention_indexed.md) | Prolog/Qnorm export, indexer projections/norm/RoPE/Hadamard, index-cache write and top-k, SFA, MLA tail | `--native-attention` binds indexed and shared-indexer backbone paths; whole-model validation remains required |
| [Routed/shared composition](../tests/glm52/routed_int4/README.md) | BF16 routed input quantization and route metadata, routed INT4 and shared W8A8 projections and activations, interleaved across AIC/AIV | `--routed-shared-kernel` connects the local body through an explicit vendor callback hook; whole-model validation remains required |
| [Complete FFN](../tests/glm52/whole_layer/ffn_original_shared_gu_static/README.md) | Residual/AddRMSNorm, router projection/top-8, routed/shared quantization and expert bodies, ordered combine and shared addition | One-launch entry; original-FFN gate passes 704 boundaries including immediate consumers across both weight sets and M1/2/12/32; TP reductions remain required |
| [Local layer chain](../tests/glm52/whole_layer/layer_local_chain/README.md) | Input residual/norm/A8, indexed or shared-indexer attention, complete NZ FFN | Eager and official ACL graph comparisons pass M1/2/12/32 at 24 core groups; each checks 3,488 boundaries. N512 paired M12 timing saves 20.22/21.21 us per local two-layer pair against three native stages |
| [Layer with TP reductions](../tests/glm52/whole_layer/layer_tp_chain/README.md) | Local layer chain with in-kernel attention-output and FFN-output AllReduce | Two-rank eager/graph gates pass; M12 paired timing saves 23.049/23.264 us per two-layer pair. A three-rank cancellation test exposes a reduction-order mismatch with FP32 HCCL; correction and TP16 acceptance remain required |

On Ascend910_9392 with CANN 9.2.0, the MLA tail matched its CANN chain numerically
at zero tolerance across 11 widths and 88 changing-input graph replays. Routed experts
matched the stock CANN ND-weight chain across eight widths and 60 changing-input,
changing-route graph replays. These are synthetic TP-shard tests, before TP
reduction. The deployed vLLM-Ascend NZ-weight comparison remains required.
The [device reports](evidence/ascend-decode/megakernel-device-20261021/README.md)
record exact versions, library/source hashes, commands and numerical scope.

Further building blocks have passed changing-input device graph comparisons:
[Prolog V3 plus SFA](../tests/glm52/whole_layer/attention.md) for layers that
reuse an existing indexer selection,
[its composition with the projection tail](../tests/glm52/whole_layer/attention_full.md),
[residual/norm/dynamic A8](../tests/glm52/whole_layer/norm.md), and the composition
from residual/norm through the attention output projection. The full attention
entries connect through `--native-attention`; input residual/norm remains
separate in that adapter. The native
[index-cache writer and selector](../tests/glm52/whole_layer/indexer_contract.md)
passes INT8 and BF16 cache comparisons at M=1,2,12,32 with 24 core groups,
including exact index order. The
[indexer front](../tests/glm52/whole_layer/indexer_front.md) also passes
changing-input comparisons at those widths, retaining the Q/K projections,
BF16 key normalization, RoPE, Hadamard, quantized outputs and head weights.
A [SHMEM probe](../tests/glm52/whole_layer/communication.md)
passes two-rank changing-input graph tests with two BF16 reductions separated
by a real CUBE projection; TP16 acceptance remains required.

The reductions now run inside the layer launch rather than outside it. A
[three-rank whole layer](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-tp-tree-v1-a3-194/AUDIT.md)
carries both FP32 tree reductions in one launch and passes eager and official
ACL graph arms on dies 4, 6 and 7: 5,424 raw byte checks and 192 tree checks
per arm at M=1 and M=12 with eight changing inputs per width, 678 unchanged
capture boundaries and no Python calls during replay. All 96 retained output
digests are identical between the two arms, so capture reproduces the eager
result rather than only passing its own oracle. No TP16 device collective
executes, the checkpoint is not loaded and neither arm is timed.

The composed indexed-attention entry also passes M=1,2,12,32 with both
INT8 and BF16 index caches, including changing main/indexer page maps and
queued producers. The two cache families use independent physical mappings.
Production boundaries, complete cache contents and ordered top-k indices match
as raw bytes. The corrected implementation's
[eight-bank comparison](evidence/ascend-decode/megakernel-device-20261021/a3-162/attention-indexed-scale-order-bank-summary.json)
uses sequence length 8192 and 280.66 MiB of distinct weight storage. At M=12/32,
graph medians are 297.03/402.49 us for INT8 index caches and 285.27/386.31 us
for BF16 index caches. The corresponding separate-stage medians are
308.67/423.17 us and 306.32/405.88 us. All 30 alternating event pairs favor
the composed launch. Each event contains 30 replays of a 16-call graph;
queued input copies are outside timing. The calls use equal-valued weights,
and numerical checks inspect the last call's retained boundaries. The
reference combines stock CANN Prolog/SFA with the independently validated
native front, selector and projection tail. TP reduction remains outside.

A separate [phase profile](evidence/ascend-decode/megakernel-device-20261021/a3-162/attention-indexed-profile-summary.json)
preserves the original synchronization and measures same-worker SYS_CNT
intervals. At M=12 with C8 index caches, maximum per-worker median windows
are 42.38 us for Prolog, 70.37 us for the indexer front, 54.84 us for the
selector/cache stage, 76.97 us for SFA and 56.55 us for the projection tail.
These maxima come from potentially different workers and are not additive.
Eight-bank graphs and 30 alternating event pairs compare the instrumented and
ordinary kernels; the C8 M=12 median paired increase is 7.26 us in this run.
Instrumentation also changes code generation and placement, so that difference
is not a fixed cost to subtract from other measurements.

A [Q/K concurrency comparison](evidence/ascend-decode/megakernel-device-20261021/a3-162/attention-indexed-parallel-bank-summary.json)
assigns the independent query and key projections to disjoint core groups.
All checked production boundaries match raw bytes. With eight weight banks,
the M=12 median paired savings are 7.16 us for C8 and 1.37 us for BF16 caches.
At M=32, C8 saves 0.91 us, while BF16 regresses by 9.98 us in all 30 pairs.
A subsequent [same-address comparison](evidence/ascend-decode/megakernel-device-20261021/a3-162/attention-indexed-parallel-placement-summary.json)
recaptures both kernels on each of two arenas. BF16 M=32 saves -0.19 us on
arena A and +0.94 us on arena B, with all production bytes preserved.
The initial regression cannot be attributed solely to the projection schedule;
the sequential schedule remains the default.

The [packed-KV candidate](evidence/ascend-decode/megakernel-device-20261021/a3-162/attention-indexed-shadow-summary.json)
combines each physical row's 512 latent and 64 RoPE values in a separate cache,
updates current slots during the indexer front, and gathers each pair with one
DMA command before restoring SFA's planar workspace. It passes 128 changing
replays across both index-cache modes, four widths and 16/24 core groups, plus
48 negative-slot replays. Complete original and shadow caches match raw bytes.
The eight-bank, same-address comparisons regress by 14.87–21.65 us at M=12
and 11.12–16.52 us at M=32, with zero wins in 240 paired comparisons.
Per-call updates are included; initial packing is outside timing. The 68-page
fixture adds 9.56 MiB per layer. This candidate is not selected.

The [packed-workspace variant](evidence/ascend-decode/megakernel-device-20261021/a3-162/attention-indexed-shadow-v2-summary.json)
also keeps the SFA merge rows packed, removing that deinterleave while
preserving the CUBE operands and reduction order. It passes the same 128
changing-input and 48 negative-slot replays. Same-address comparisons still
regress by 14.05–20.00 us at M=12 and 25.93–37.33 us at M=32, with zero wins
in 240 pairs. The sequential attention implementation remains selected.

The corrected attention adapter comparisons use original
[FP32](evidence/ascend-decode/megakernel-device-20261021/a3-162/attention-adapter-fp32-scale-order.json)
and [BF16](evidence/ascend-decode/megakernel-device-20261021/a3-162/attention-adapter-bf16-scale-order.json)
projection scales with the production body, indexer packing and workspace
pool. Both indexer modes pass M=1,2,12,32 with eight changing graph replays
at 24 groups. The test
uses independent INT32 main/indexer slot sources and page maps, a separate
warmup stream, and reserved capture storage. It copies ordered top-k to
persistent storage and feeds a second shared-indexer call with the same
hidden/weights/cache. All retained production boundaries match raw bytes.
This validates those local calls and storage lifetimes; it does not execute
the vLLM forward hook, ACLGraphWrapper or TP reduction.

The [real-class installation gate](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-install-v6/rank-00.json)
passes on one A3 die with the pinned vLLM/vLLM-Ascend packages and Torch NPU
2.10. It constructs the actual TP16 attention shard with four local heads,
runs official weight postprocessing and cache binding, loads both native
libraries and installs the C8 attention adapter. Fifteen Gloo helper ranks
remain CPU-only, and all sixteen processes exit successfully. This gate's
scope ends at adapter installation; layer forward and graph execution are
separate acceptance steps.

The [real outer-forward comparison](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-forward-m1-v9/outer-forward.json)
passes C8 M=1 at an 8192-token context through the actual layer,
`mla_forward` and `ForwardContext`. Final BF16 output, the complete main and
indexer caches, indexer scales and ordered top-k match the original forward
byte for byte. The installed call executes the native body exactly once.
The adapter normalizes RoPE metadata to ND before calling the native body;
the source indexer metadata can have physical format NCHW after a 4D clone
or official builder allocation. This comparison uses synthetic weights and
one eager TP16 local shard before reduction. A [wider C8 run](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-forward-c8-v9/outer-forward.json) passes eight
inputs each at M=1 and M=2, then finds two different BF16 output elements
at M=12 on its second input, with maximum absolute difference
1.52587890625e-5. All caches and top-k still match exactly. That difference
is reproduced with identical inputs and output bytes on a3-194. The
[retained-stage diagnostic](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-forward-diagnostic-v10-a3-194-summary.json)
finds identical Q, SFA output, value-up, quantization codes/scales and O
INT32 products. The final O epilogue is the first differing boundary:
`accumulator * (channel * token)` reconstructs every vendor output byte,
while `(accumulator * channel) * token` reconstructs every native output
byte. The [two exact witnesses](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-o-cpu-witness-v1.json)
lie on BF16 rounding boundaries. The installed
[CANN source](evidence/ascend-decode/megakernel-device-20261021/integration/o-epilogue-source-audit/README.md)
contains both arithmetic orders in different QMM templates; scale dtype
alone does not select the required order. The
[combined-scale candidate](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-forward-o-combined-v11-m12-a3-194/outer-forward.json)
passes both M=12 inputs, including the original failure. Its
[wider C8 comparison](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-forward-o-combined-v11-int8-wide-a3-194/outer-forward.json)
passes all eight inputs at M=1,2,12 and the first five at M=32. The sixth
M=32 input exposes two differing INT8 index-cache codes and 45 differing
ordered top-k indices; O output, main caches and index scales still match
raw bytes. The [BF16 indexer run](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-forward-o-combined-v11-bf16-wide-a3-194/outer-forward.json)
fails at the same M=32 input with two differing BF16 cache elements and 52
different top-k indices. Both runs pass the preceding 29 cases. The
[indexer diagnostic](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-index-diagnostic-v13-summary.json)
reproduces both failures byte for byte and finds the same four differing
BF16 WK projection elements. All query stages and projection operands match.
One difference is the row-6 attention weight at channel 25, preceding that
row's top-k change. Each arm's RoPE agrees with an independent calculation
from its own BF16 input. The native WK outputs also agree with rounding
their retained FP32 accumulators; the remaining discrepancy concerns the
projection accumulation path. An auxiliary FP64 dot disagrees with the
vendor and native outputs at different coordinates, so higher dot precision
alone cannot reproduce the production result.

The [captured WK geometry comparison](evidence/ascend-decode/megakernel-device-20261021/integration/captured-wk-v2-a3-194/summary.json)
repeats seven stock BF16-output paths three times on that same M=32 input.
All four full-N=160 paths reproduce the complete vendor projection, while
all three paths split into five N=32 products reproduce the complete native
projection. ND/NZ storage and transposed/contiguous operands do not change
either group. This isolates column decomposition as a reproducible
discriminator for the captured input. A subsequent
[native full-width probe](evidence/ascend-decode/megakernel-device-20261021/integration/captured-wk-native-v1-a3-194/summary.json)
changes N32 to N160, baseK from 512 to 96, and the loading steps and iteration
order. Both native arms nevertheless reproduce all 5,120 original FP32
accumulators exactly across three repeats, retaining the same four BF16
differences. Native column width and these tiling changes are therefore
insufficient to reproduce the stock full-width path.
The [stock FP32-output comparison](evidence/ascend-decode/megakernel-device-20261021/integration/captured-wk-fp32-v1-a3-194/summary.json)
first reproduces all seven BF16 control paths under Torch NPU 2.12, then
executes full-N160 and split-N32 FP32-output matrix multiplies. Both return
all 5,120 native accumulators exactly across three repeats. Casting the
full-width FP32 result to BF16 retains the same four differences from the
direct BF16 output. Output dtype can select a different kernel or reduction
path, so this observation does not isolate the final cast as the cause.

The [dispatch and binary audit](evidence/ascend-decode/megakernel-device-20261021/integration/captured-wk-dispatch-source-audit/dispatch-schema.json)
identifies a MatMulV2 BF16 path that changes K traversal with the N tile.
For the captured M32/N160/K6144 call, its 24 K256 segments are shifted and
alternately reversed across N16 tiles. An isolated
[single-L0C reproduction](evidence/ascend-decode/megakernel-device-20261021/integration/captured-wk-kshift-v1-a3-194/summary.json)
passes three alternating repeats: natural K order reproduces all 5,120
native FP32 accumulators and the four BF16 differences; the vendor K order
reproduces all 5,120 vendor BF16 outputs. Both direct BF16 stores equal the
cast of their own FP32 accumulators. This establishes an accumulation-order
mechanism for this captured call.

The [M32 candidate](../tests/glm52/whole_layer/indexer_wk_kshift/README.md)
preserves the packed N32 weight/cache layout and uses the vendor K order in
20 M16/N16 cube tasks. The actual outer-call gates pass eight changing
inputs in both [BF16](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-wk-kshift-v14-bf16-m32-a3-194/summary.json)
and [C8](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-wk-kshift-v14-int8-m32-a3-194/summary.json),
including the previously failing sixth input: all 88 output, ordered-top-k
and cache boundary checks match raw bytes. The final input also passes all
19 BF16 and 23 C8 indexer diagnostic boundaries.

The same compiled candidate passes the official graph wrapper at M=1,2,12,32
in [BF16](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-wk-kshift-v15-bf16-all-graph-a3-194/summary.json)
and [C8](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-wk-kshift-v15-int8-all-graph-a3-194/summary.json):
16 captures, 128 graph replays and 704 raw-byte comparisons against the
original eager outer call all pass, with no Python body calls during replay.
Inputs and metadata change across eight replays per width. These gates cover
the synthetic TP16 local shard; shared-indexer outer calls, TP collectives
and whole-model timing remain separate gates.

The [Torch NPU 2.12 runtime](evidence/ascend-decode/megakernel-device-20261021/integration/runtime212-build-v1/attention/manifest.json)
rebuilds the host bindings while retaining the verified CANN device objects.
The actual layer-6 graph gates for
[C8](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-runtime212-layer6-int8-graph-v3-a3-194/acceptance-summary.json)
and [BF16](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-runtime212-layer6-bf16-graph-v3-a3-194/acceptance-summary.json)
pass M=1,2,12,32 with eight changing inputs per width: 16 captures,
128 graph replays and 704 raw-byte checks. The original constructor selects
the full indexer in this MoE layer. Sixteen real Gloo ranks complete; only
rank zero initializes the one exposed NPU. This verifies attention in the
same Torch runtime required by the FP32 gate projection. It excludes the
FFN and TP reductions. The host launcher requires 64 GiB of memory and a
writable temporary working directory for the official runtime report.

The independent [shared-attention descriptor](../tests/glm52/whole_layer/attention_shared_descriptor/README.md)
passes all eight M=1,2,12,32 and 16/24-group numerical configurations.
Its [M12/24 same-address comparison](evidence/ascend-decode/megakernel-device-20261021/integration/shared-attention-descriptor212-timing-v1-a3-194/summary.json)
regresses by 2.201/3.022 us in the two owner arenas, with zero wins in
60 pairs. All 1,152 per-bank and 480 event-tail raw-byte checks pass.
The descriptor change is not selected.

The independent [shared-attention terminal variant](../tests/glm52/whole_layer/attention_terminal/README.md)
keeps the default callable join and changes only the standalone global
entry's final completion to local PIPE_ALL, DDR and ordinary pipe destruction.
Its [eight-geometry gate](evidence/ascend-decode/megakernel-device-20261021/integration/shared-attention-terminal212-wide-v1-a3-194/summary.json)
passes 384 raw-byte checks, including 128 immediate consumers inside the same
graph. At M12/24 and sequence length 8192, its
[same-address comparison](evidence/ascend-decode/megakernel-device-20261021/integration/shared-attention-terminal212-timing-v1-a3-194/summary.json)
saves median 1.233/0.733 us, with 26 of 30 pairs faster in each owner arena.
All 1,152 numerical and 480 event-tail checks pass. This small local gain
has not been measured in the actual engine.

The [actual DecoderLayer construction gate](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-decoder212-constructor-v2-a3-194/acceptance-summary.json)
constructs layers 6 and 7 and completes official weight postprocessing with
TP16 shard dimensions on one NPU. All sixteen processes exit successfully.
The original W4A8 constructor uses a real singleton HCCL group for communicator
metadata; no TP16 collective or layer forward executes. Routed weights have
the expected packed INT4 NZ representation. This fixture leaves shared
gate/up in BF16 because direct construction omitted the official loader's
packed-module mapping; it does not pass the target shared W8A8 contract.

The [full-decoder forward fixture](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-decoder212-forward-int8-m1-v1-a3-194/failure-summary.json)
uses the official loader mapping and passes both layers' shared W8A8/NZ
contracts, selecting `A8_INT_FUSED`. Both indexers retain the same NPU top-k
owner. The original layer-6 forward fails with error 507015 before the native
arm executes. The driver identifies routed `GroupedMatmulSwigluQuantV2`,
stream 47/task 144, as the failing task. This fixture uses synthetic parameters and explicit local TP
boundary cuts.

The [synchronized GMM1 diagnostic](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-decoder212-diagnostic-int8-m1-v1-a3-194/failure-summary.json)
completes the pre-call synchronization and confirms 256 nonnegative expert
counts summing to the eight input rows. GMM1 submission returns, then its
completion synchronization fails before `dispose_tensor` executes. Invalid
counts and that later input disposal are therefore unnecessary to reproduce
this failure. Original operator arguments, code and exception dumps are
retained for packed INT4/NZ and workspace inspection.

The [exception decode](evidence/ascend-decode/megakernel-device-20261021/integration/decoder-nz-descriptor-audit-v1/README.md)
records an INT4/NZ physical descriptor spanning 805,306,368 bytes over a
201,326,592-byte allocation, with actual tiling N=1024 instead of 256.
The INT32 carrier retained its INT8 owner's physical dimension units.
An isolated host alias supplies a canonical INT32 storage descriptor while
retaining the original allocation and owner. The
[corrected M1 diagnostic](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-decoder212-storage-int8-m1-v1-a3-194/summary.json)
preserves all raw bytes in both layers' four weight banks, completes all four
original GMM1 calls, and passes 25 raw-byte comparisons across layers 6 and 7.
All sixteen Gloo ranks complete; only rank zero uses the NPU. Both arms use
the official FFN, with native attention compared against original attention.
The [normal asynchronous M1 run](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-decoder212-storage-forward-int8-m1-v2-a3-194/summary.json)
also passes all 25 comparisons with the original disposal timing and no
diagnostic GMM1 fences. The
[changing-input extension](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-decoder212-storage-forward-int8-wide-v2-a3-194/failure-summary.json)
passes eight cases each at M1 and M2, then fails at M12: one index-cache INT8
element differs by one and five top-k entries differ in each layer's shared
owner. Both layers' outputs and residuals remain byte-equal. The run stops at
this boundary before M32. Full-layer graph replay remains a separate gate;
both forward runs retain explicit local TP cuts.
The [full-decoder indexer diagnostic](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-decoder212-index-int8-m12-v1-a3-194/summary.json)
reproduces every saved failure boundary byte for byte. The first difference
is in two BF16 key-projection elements; it propagates through key
normalization, RoPE and Hadamard to the single INT8 key element. Inputs,
weights, the query path and scales agree exactly. The
[actual M12 dispatch audit](evidence/ascend-decode/megakernel-device-20261021/integration/captured-wk-m12-source-audit/README.md)
identifies twelve K512 segments, with K128 MMAD steps, in one L0C accumulator.
The [captured native reproduction](evidence/ascend-decode/megakernel-device-20261021/integration/captured-wk-m12-kshift-v1-a3-194/summary.json)
matches all 1,920 original BF16 values across three interleaved repeats after
applying that segment order. Its natural-order control reproduces the saved
native FP32 and BF16 values, and all four guard rows remain unchanged.
The M12 branch in indexed-attention library `5f8868bf` passes the
[complete DecoderLayer comparison](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-decoder212-wk-m12-int8-wide-v1-a3-194/summary.json):
M1/M2/M12/M32, eight changing inputs each, both original layer weight sets,
and all 816 raw-byte boundaries. Both arms retain the original FFN; the two
TP reductions are explicit local cuts.

The [whole-Decoder graph comparison](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-decoder212-graph-int8-m12-v1-a3-194/summary.json)
captures the original DecoderLayer 6-to-7 chain through ACLGraphWrapper at M12,
with absent and provided initial residuals. Four captures, eight changing-input
replays and all 204 raw-byte boundaries pass. Both arms retain the original
FFN. Capture claims reserved native workspace; replay invokes no Python
forward calls, and observations are poisoned before each replay.
The [expanded graph input set](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-decoder212-graph-int8-wide-v1-a3-194/failure-summary.json)
passes eight M1 inputs without an initial residual and two with one. The third
provided-residual input has one differing INT8 index-cache element in the
native arm; the vendor graph, all other native boundaries, ordered top-k and
outputs match. It records 555 passing boundaries out of 556 before stopping.
The [retained-indexer diagnostic](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-decoder212-graph-index-int8-m1-v3-a3-194/summary.json)
reproduces all 60 saved plan and output tensors and the recorded live input
identities. Both arms' retained key codes and scales equal the bytes written
to the current cache slot. WK is the first differing recorded boundary:
at `[0,25]`, the original BF16 projection is -0.3359375 and native is
-0.333984375. The downstream INT8 codes at channel 46 are 20 and 21.
Projection operands and all query stages match. Each RoPE result matches the
FP32 formula applied to its own input. This diagnostic preserves the original
graph failure.

The [captured M1 K-order experiment](evidence/ascend-decode/megakernel-device-20261021/integration/captured-wk-m1-kshift-v3-a3-194/summary.json)
uses the actual MatMulV2 key98513 dispatch's N16 tiles and K512 traversal,
with 48 K128 updates to one FP32 accumulator tile. The M1 A operand uses the
A3 GEMV layout. Across three alternating repeats, natural K order reproduces
all 160 captured native FP32 accumulators and BF16 outputs; the vendor K order
matches all 160 vendor BF16 outputs. At `[0,25]`, the latter accumulator is
exactly -0.3349609375, the BF16 midpoint, which rounds to -0.3359375.
All fifteen guard rows remain unchanged. The [independent raw-output audit](evidence/ascend-decode/megakernel-device-20261021/integration/captured-wk-m1-kshift-v3-a3-194/root-raw-audit.json)
confirms all six retained calls. Indexed-attention library `8deff750` integrates
this M1 branch without changing M2/M12/M32. It passes the [original Decoder
eager gate](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-decoder212-wk-m1-int8-eager-wide-v1-a3-194/summary.json)
at all four widths with eight changing inputs each: all 816 boundaries match.
The same library passes the [expanded original Decoder graph gate](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-decoder212-wk-m1-int8-graph-wide-v1-a3-194/summary.json):
16 captures, 128 replays and 3,264 raw-byte checks at M1/2/12/32, with eight
changing inputs for each residual entry mode. Actual sequence lengths and
independent main/indexer page maps change between replays; the native capture
claims its reserved buffers, and replay enters neither Python body. All four
NZ weight banks retain their original allocations and bytes. Both arms retain
the original FFN and explicit local TP cuts. Sixteen Gloo ranks complete,
with only rank zero using the single exposed die.

The [complete native Decoder graph comparison](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-decoder212-native-full-graph-wide-v1-a3-194/summary.json)
also replaces each layer's post-attention norm and MLP with complete FFN
`381febf9`, alongside indexed attention `8deff750` and shared attention
`64108f6f`. All 16 captures, 128 changing-input replays and 3,264 raw-byte
checks pass against the original two-layer chain. Each native capture claims
separate layer-owned FFN storage for its exact width and actual stream. The
adapter performs one original MoE runner lookup and retains its final-reduction
and zero-expert helpers; replay invokes no Python adapter. Four original NZ
banks keep their allocation addresses and raw bytes. The two TP cuts remain
external post-AR1 input and local pre-AR2 output; this gate uses one die.

The [single-launch local layer build](evidence/ascend-decode/megakernel-device-20261021/integration/layer-local-chain-build-v4/summary.json)
composes input AddRMSNorm/A8, attention and complete FFN under one mixed entry.
Both indexed and shared-indexer variants compile and link with CANN 9.2;
Torch212 loads the library without initializing the NPU. Eight ARM metadata
contracts check the 1,600-byte kernel argument record. The host shim submits
queued producers through the current TorchNPU stream. Its local attention
partial feeds FFN directly with identity TP cuts; device numerical and graph
comparisons remain required.

The separate [actual FFN comparison](../tests/glm52/whole_layer/vendor_hook_ffn_extension/README.md)
uses each original layer's post-attention norm, router, shared expert and routed
GMM1/GMM2 against complete native library `8430ef17`, with the same corrected
NZ allocations and original scale bit patterns. Both layer weight sets pass
all fourteen M1 boundary checks. The changing-input extension passes eight
inputs each at M1/M2/M12 and two at M32 for layer 6, then finds two shared BF16
differences at row 20 in M32 input 2. All six other boundaries, including routed,
combined and pre-AR2 outputs, remain byte-equal. It records 188 passing checks
out of 189 before stopping; layer 7's wider inputs remain untested. These are
synthetic-parameter FFN tests with explicit post-AR1 inputs and local pre-AR2
outputs, using native graph replay and the original eager operator chain.
The [shared-stage diagnostic](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-decoder212-ffn-shared-m32-v1-a3-194/summary.json)
reproduces all sixteen saved input/output tensors and locates the first
difference in three FP32 middle scales, each one ULP lower in native output.
Input quantization, GU INT32 values and middle INT8 codes are byte-equal;
both integer projections agree with exact CPU integer dots. Replacing only
those scales in the combined-scales Down epilogue reproduces every original
BF16 output. The [installed dispatch audit](evidence/ascend-decode/megakernel-device-20261021/integration/shared-activation-scale-source-audit/README.md)
identifies `DequantSwigluQuant_int32_high_performance_30013`. This implementation
uses scalar `amax / 127`, a guarded reciprocal, vector multiplication, and
INT16/half/INT8 casts. The native grouped path instead multiplies amax by a
rounded reciprocal and divides each value by the scale. Both scale and code
arithmetic change in the [original-shared candidate](../tests/glm52/whole_layer/ffn_original_shared/README.md).
Library `342d9653` passes the
[original-FFN device gate](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-decoder212-ffn-original-shared-wide-v1-a3-194/summary.json):
both layer weight sets, M1/M2/M12/M32, eight changing inputs each, and all
448 raw-byte boundaries, including the previously failing shared branch.
Native FFN graph replay is compared with the original eager norm/MLP chain;
attention and multi-rank reductions remain outside this gate.
The [composition with token-owned splitting and terminal completion](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-decoder212-ffn-original-shared-terminal-wide-v1-a3-194/summary.json),
library `c39acf92`, passes the same 64 cases and all 448 arithmetic boundaries.
It also passes 256 checks of immediate combined, routed, shared and retained
residual consumers in the captured graph. Consumer storage is allocated before
capture and poisoned before every replay. All four original NZ weight banks
retain their addresses and bytes; only rank zero initializes the NPU.
Its [paired 16-bank graph benchmark](evidence/ascend-decode/megakernel-device-20261021/integration/ffn-original-shared-terminal-timing-v2-a3-194/summary.json)
compares against `342d9653`, with the same corrected shared arithmetic in both
arms. At M12/24 and 65 active experts, candidate/parent medians are
211.408/219.639 us and 211.261/221.010 us across two independently allocated
arenas. Median paired savings are 8.239/9.763 us, with all 60 pairs faster.
All 19,200 numerical, 360 event-tail and 696 active-table checks pass.

The [static-GU composition](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-decoder212-ffn-original-shared-gu-static-wide-v1-a3-194/summary.json),
library `381febf9`, passes the same 64 original-FFN cases and all 704 checks.
Its [matched-address comparison](evidence/ascend-decode/megakernel-device-20261021/integration/ffn-original-shared-gu-static-timing-v1-a3-194/summary.json)
retains the corrected shared arithmetic in both arms. At M12/24 with 65 active
experts and 16 weight banks, candidate/`c39acf92` medians are
202.559/210.716 us and 200.956/211.412 us. Median paired savings are
8.139/10.350 us, with all 60 pairs faster and all 19,200 numerical,
360 event-tail and 696 active-table checks passing.

The independent [expert-ready GU candidate](../tests/glm52/whole_layer/ffn_gu_ready/README.md)
starts activation on the paired AIVs after each complete expert GU product.
Four publication credits bound the producer lead; both AIVs acknowledge each
expert, and all credits drain before the unchanged second body join. Fewer
active experts than AIC groups select the original balanced activation path.
Library `cd0b54dd` passes the same 64 original-FFN cases and all 704 checks.
Its [same-address timing](evidence/ascend-decode/megakernel-device-20261021/integration/ffn-gu-ready-timing-v1-a3-194/summary.json)
against `c39acf92` saves 7.317/7.495 us at M12/24, 65 active experts and
16 weight banks. All 60 pairs favor the candidate; all 19,200 numerical,
360 event-tail and 696 active-table checks pass. This comparison retains
dynamic GU tiling in both arms.

The [static-GU plus expert-ready composition](../tests/glm52/whole_layer/ffn_gu_static_ready/README.md),
library `affc88ba`, also passes all 64 original-FFN cases and 704 checks.
Its [paired comparison with static-GU parent `381febf9`](evidence/ascend-decode/megakernel-device-20261021/integration/ffn-gu-static-ready-timing-v1-a3-194/summary.json)
regresses by 6.006/2.449 us across the two arenas, with no wins in 60 pairs.
Candidate/parent medians are 207.067/201.114 us and 208.213/205.726 us;
all 19,200 numerical, 360 event-tail and 696 active-table checks pass.
The composition is not selected. These independent gains do not combine in
this schedule; the regression's cause has not been isolated.

Two further candidates retain the static-GU parent. [Down N512](../tests/glm52/whole_layer/ffn_down_n512/README.md),
library `c90b3666`, changes only the Down host tiling from 24 N256 output tiles
to 12 N512 tiles. The original-FFN gate passes all 704 checks; 72 full 200-byte
checks verify the actual prepared tiling before capture and after each replay.
Its [paired comparison](evidence/ascend-decode/megakernel-device-20261021/integration/ffn-down-n512-timing-v1-a3-194/summary.json)
against `381febf9` saves 10.418/8.867 us, with all 60 pairs faster. Candidate/
parent medians are 201.130/211.521 us and 200.005/209.023 us. All 19,200
numerical, 360 event-tail and 696 active-table checks pass. Within each pair,
32 scratch tensors share addresses; each arm retains its own immutable Down
tiling. Twenty full 200-byte checks confirm these tilings throughout the run.
[Norm publication](../tests/glm52/whole_layer/ffn_norm_join/README.md), library
`ddd2f7b9`, removes only the private norm helper's final mixed-core join while
retaining its caller's immediate publication and join. It also passes all 704
checks. Its [same-address timing against `381febf9`](evidence/ascend-decode/megakernel-device-20261021/integration/ffn-norm-join-timing-v1-a3-194/summary.json)
regresses by 0.953/0.475 us, with no wins in 60 pairs. All numerical,
event-tail and active-table checks pass. This candidate is not selected;
the original norm join is retained.

The [Down M32/N1024/K64 candidate](../tests/glm52/whole_layer/ffn_down_n1024/README.md),
library `bbd52e42`, changes only the N512 parent's host tiling. For at most
16 tokens per expert, it halves output-tile copies while keeping the MMAD
count unchanged. Its [original-Decoder FFN comparison](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-decoder212-ffn-down-n1024-wide-v1-a3-194/summary.json)
passes M1/2/12/32 at 24 groups with two synthetic weight sets and eight changing
inputs: 704 byte checks, including 256 immediate-consumer checks, and 72 full
Down-tiling checks. Its [paired M12/24 comparison](evidence/ascend-decode/megakernel-device-20261021/integration/ffn-down-n1024-timing-v1-a3-194/summary.json)
against N512 saves 7.933/6.371 us across two arenas, with all 60 pairs faster.
Candidate/control medians are 201.439/209.362 and 199.486/205.869 us with 65
active experts and 16 weight banks. All 19,200 numerical, 360 event-tail,
696 active-table and 20 full-tiling checks pass. The [hot-expert gate](../tests/glm52/whole_layer/ffn_down_n1024/hot/README.md)
passes M17/32 at both 16 and 24 groups, forcing all tokens onto the same eight
experts and exercising both Down M32 tiles. Each run passes 480 byte checks,
including 128 immediate-consumer and 128 routing-metadata checks, and 36 full
tiling checks. The [609 local-layer composition](../tests/glm52/whole_layer/layer_down_n1024/README.md)
retains the original Top-K copy and passes eager and official graph gates at
M1/2/12/32: each passes 3,488 byte comparisons and 136 full-tiling checks.
Its [matched whole-layer timing](evidence/ascend-decode/megakernel-device-20261021/integration/layer-down-n1024-timing-v1-a3-194/summary.json)
saves 1.435/2.487 us per indexed/shared two-layer pair against N512, with
51 of 60 pairs faster. Both arms use the same 609 binary and production
addresses; only the two Down-tiling pointers differ. All 6,660 numerical,
600 event-tail, 844 complete-tiling and 388 complete-record comparisons pass.
The one-bank whole-layer effect is smaller than the 16-bank standalone FFN
effect; both local-layer reductions remain identities.

An installed-CANN [host/compiler probe](evidence/ascend-decode/megakernel-device-20261021/integration/ffn-down-static-tiling-probe-v2/README.md)
derives an M32/N1024/K128 single-buffer Down plan. Six named tiling fields
change; L1/L0B/L0C fit the same allocation limits. K128 would halve MMAD calls
relative to K64, while `dbL0B=1` also disables A/B ping-pong in this path.
The complete 200-byte proposal is identical for ND and NZ. The
[complete candidate](../tests/glm52/whole_layer/ffn_down_k128/README.md), library
`0842cd0f`, now passes ARM compilation, actual Torch 2.12 co-loading, both
host tilers and linked planner-POD checks. Its 104 source files and seven
artifacts are pinned. The [original Decoder FFN gate](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-decoder212-ffn-down-k128-wide-v1-a3-194/summary.json)
passes M1/2/12/32 with two synthetic weight sets and eight changing inputs:
704 byte comparisons, including 256 immediate-consumer checks, and 72 complete
Down-tiling checks. Its [hot-expert gate](../tests/glm52/whole_layer/ffn_down_k128/hot/README.md)
also passes M17/32 under both 16 and 24 groups, exercising both M32 tiles.
Each configuration passes 480 byte comparisons and 36 complete Down records.
Its [same-address paired timing](evidence/ascend-decode/megakernel-device-20261021/integration/ffn-down-k128-timing-v1-a3-194/summary.json)
saves 7.339/5.872 us against N1024/K64, with all 60 pairs faster.
Candidate/control medians are 203.950/211.309 and 203.187/209.089 us at
M12/24, 65 active experts and 16 weight banks. All 19,200 numerical,
360 event-tail, 696 active-table and 20 complete-tiling checks pass.
The [matched whole-layer comparison against N512](evidence/ascend-decode/megakernel-device-20261021/integration/layer-down-k128-timing-v1-a3-194/AUDIT.md)
does not establish a consistent benefit. Paired median savings are
-1.142/+0.743 us per indexed/shared pair, with 6/30 and 23/30 wins across
the two arenas. All 6,660 numerical, 600 event-tail, 844 complete-tiling and
388 complete-record comparisons pass. Both arms use the same 609 binary,
addresses and Top-K copy; only the two Down-tiling pointers differ.
K128 is not selected for the local layer. Its 16-bank standalone improvement
does not carry over consistently to this one-bank layer workload.

The [dual input-quantization candidate](../tests/glm52/whole_layer/ffn_dual_quant/DESIGN.md),
library `715060c0`, shares the BF16 load, FP32 widening and maximum-absolute-value
reduction while retaining the distinct routed and shared quantization suffixes.
Its [direct quantizer comparison](evidence/ascend-decode/megakernel-device-20261021/integration/ffn-dual-quant-probe-v1-a3-194/summary.json)
passes all eight M1/2/12/32 by 16/24-group geometries: 384 graph replays and
3,456 raw-byte checks, including immediate consumers and two different output
poisons. Inputs include signed zeros, finite extremes, Inf, NaN payloads and
a BF16 value whose routed/shared scales differ by one FP32 ULP. The
[complete-FFN comparison](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-decoder212-ffn-dual-quant-wide-v1-a3-194/summary.json)
also passes against the original decoder norm and MLP: 64 cases, 704 raw-byte
checks and 72 full Down-tiling checks across M1/2/12/32 and two synthetic
weight sets. Its [matched timing against the N512 parent](evidence/ascend-decode/megakernel-device-20261021/integration/ffn-dual-quant-timing-v1-a3-194/summary.json)
regresses by 3.652/2.182 us across two independently allocated arenas, with
no wins in 60 pairs. All 19,200 numerical, 360 event-tail and 696 active-table
checks pass. The N512 parent is retained.

The [official ACLGraphWrapper gate](evidence/ascend-decode/megakernel-device-20261021/integration/vendor-hook-graph-v12-summary.json)
passes C8 and BF16 at M=1,2,12 with eight changing inputs per width, plus
the initial two-input C8 M=12 probe. All 14 real graph captures, 100 graph
replays and 552 raw-byte boundary comparisons pass against the original
eager outer call. Input storage remains fixed while actual sequence lengths,
page tables, separate main/index slots, RoPE values, hidden states and cache
contents change. The native capture claims preallocated workspace and both
slot casts on its actual capture stream. Replays execute no Python body;
distinct nonzero output hashes match the eager reference on every input.
This earlier gate validates the local-shard graph path at M=1,2,12 with
synthetic weights; the K-order candidate above extends coverage to M=32.

Extended
[FP32 checks](evidence/ascend-decode/megakernel-device-20261021/a3-162/indexer-front-fp32-extended-control.json)
and the [BF16 scale probe](evidence/ascend-decode/megakernel-device-20261021/a3-162/indexer-front-bf16-scales.json)
expose rare projection rounding differences in the preceding WQ implementation. For this
K=2048, N=4096 projection, stock QMM combines the token and channel scales
before multiplying the INT32 accumulator. The corrected front preserves the
original scale dtype and uses that order for both FP32 and BF16 scales.
[FP32](evidence/ascend-decode/megakernel-device-20261021/a3-162/indexer-front-fp32-scale-order.json)
and [BF16](evidence/ascend-decode/megakernel-device-20261021/a3-162/indexer-front-bf16-scale-order.json)
checks each pass all four widths, both indexer modes and 32 changing graph
replays per case. All production boundaries match raw bytes; internal FP32
normalization statistics remain diagnostic-only. The independent
[BF16 output-scale probe](evidence/ascend-decode/megakernel-device-20261021/a3-162/mla-tail-bf16-scales.json)
passes all four widths and eight changing replays with varied non-power-of-two
channel scales. Those inputs did not expose the O-projection discrepancy
subsequently found by the real outer-forward diagnostic above.

The [a3-84 evidence](evidence/ascend-decode/megakernel-device-20261021/a3-84/README.md)
includes raw-byte checks and paired timings. The batched MLA output epilogue
takes 34.25 us versus the same-run CANN chain's 37.79 us at M=12, and
39.85 versus 39.31 us at M=32. These single-bank graphs include an identical
queued input clone. Composed Prolog/SFA/projection takes 148.95 and 214.98 us
at M=12/32 versus 167.18 and 254.49 us for separate attention and the original
native tail. Adding residual/norm to that launch loses 3.90 us at M=12.
The INT32-partial routed dispatch-metadata entry includes
dispatch, metadata and row recovery: M=12 takes 182.10 us versus 294.67 us for
the original native chain; M=32 takes 307.80 us versus 461.73 us. These are
synthetic single-bank comparisons. Against a stock CANN reference using the
same canonical NZ weights, the complete native chain takes 179.33 versus
265.95 us at M=12, and 303.34 versus 354.78 us at M=32. The deployed custom-op
and whole-model measurements remain required.

An independent routed variant moves each partial's FP16 scaling to CUBE
Fixpipe, preserving the later FP32 reconstruction and BF16 expert-output
boundary. M=1,12,32 changing-route comparisons against the canonical stock
NZ chain pass byte for byte. At M=32 it removes 6.25 MiB of scratch and
12.5 MiB of partial-result read/write traffic per call. Four-bank, six-order
comparisons measure 197.26/345.10 us at M=12/32 versus 183.62/334.03 us for
the original INT32 entry and 188.28/336.98 us for a rebuilt INT32 control.
The FP16 variant is not selected for performance integration.

Redistributing AIV work by global route row preserves the INT32 products and
CUBE expert assignment. Against the original and a same-source INT32 control,
six-order four-bank comparisons measure 176.21 us at M=12 versus
181.75/180.57 us, and 292.14 us at M=32 versus 334.95/335.58 us.
Final output and dispatch metadata pass M=1,12,32 changing-route byte comparisons.
A separate gate/up `CFG_MDL` variant groups K-axis loads while retaining the
same tiling and arithmetic. Its four-bank medians are 172.01/288.80 us at
M=12/32 versus 173.98/289.26 us for a same-run balanced control. The M=12
difference has the same sign in all six orders; M=32 includes one reversal.
The variant passes all four widths and eight changing-route byte checks but
is not combined with the routed/shared entry.
An independent true-M16 padding variant also passes strict replay comparisons
but is slower in all six orders: 175.20/300.00 us at M=12/32 versus
174.64/291.78 us for its balanced control. It is not selected.

The [native BF16 input front](evidence/ascend-decode/megakernel-device-20261021/a3-162/routed-input-front-numerics.json)
adds input quantization, expert counts and stable route metadata to the routed
launch, replacing external InitRouting and dispatch recovery. It retains the
balanced INT32 projection body and original-slot combine order. M=1,2,12,32
each pass eight changing graph replays with exact bytes at all 17 checked
boundaries, including the queued-producer check. Its performance comparison
uses four independently allocated weight banks and 16 calls per graph, with
30 event samples per arm covering all six three-arm orders. At
[M=12](evidence/ascend-decode/megakernel-device-20261021/a3-162/routed-input-front-m12-bank4.json),
the median is 150.89 us versus 172.97 us for balanced dispatch and 261.28 us
for the CANN chain. At
[M=32](evidence/ascend-decode/megakernel-device-20261021/a3-162/routed-input-front-m32-bank4.json),
the medians are 269.87/285.95/353.88 us. Each bank passes all numerical
boundaries independently. These routed-only costs exclude the shared branch;
the integrated composition is measured below.

The [shared W8A8 prototype](../tests/glm52/shared_w8/README.md) covers gate/up,
SwiGLU/requantization and local down projection in one mixed launch. Its stock
CANN mode-0 comparison passes M=1,2,12,32 with eight changing-input replays.
Standalone medians of 40.89/43.06 us at M=12/32 exceed the reference's
38.98/40.65 us; composition must establish any benefit from overlapping work
or removing boundaries. A separate grouped-quantization variant preserves
the unmodified vLLM SwiGLU source's vector scaling and rounding order, and
passes all four widths and eight changing replays between stock CANN
projections.

An interleaved routed/shared entry places shared CUBE projections beside routed
AIV stages, and shared activation/dequantization beside routed CUBE stages.
It reuses the routed stage barriers and one Matmul system workspace. The
[single-die comparison](evidence/ascend-decode/megakernel-device-20261021/a3-162/README.md)
passes all retained boundaries at M=1,2,12,32 with 16 and 24 core groups,
including signed zero and zero route probabilities. It saves 16 MiB of scratch
against separate native execution. Its two BF16 outputs still precede their
addition, routed scaling and TP reduction.
With four weight banks, its six-order medians are 187.17/312.36 us at M=12/32,
versus 219.24/347.33 us for separate native stages and 300.44/427.04 us for
the CANN chain. Each arm includes the independent input quantizers and dispatch.
The TileRT adapter's body, shared-weight packing and capture workspace pool
also pass all four widths and eight changing graph replays on that die.
This device test excludes the vendor runner, top-k selection and TP finalize.

The selected routed/shared entry also incorporates the native BF16 routed
input front. Independent routed and shared inputs pass all 37 retained
boundaries at M=1,2,12,32 with 16 and 24 groups. The production adapter's
[FP32](evidence/ascend-decode/megakernel-device-20261021/a3-162/routed-pipeline-front-adapter-fp32.json)
and [BF16 shared-scale checks](evidence/ascend-decode/megakernel-device-20261021/a3-162/routed-pipeline-front-adapter-bf16.json)
also pass all widths and eight changing replays, including queued INT64
route-ID conversion. Four-bank, 16-call comparisons measure
[161.11 us at M=12](evidence/ascend-decode/megakernel-device-20261021/a3-162/routed-pipeline-front-m12-bank4.json)
and [285.15 us at M=32](evidence/ascend-decode/megakernel-device-20261021/a3-162/routed-pipeline-front-m32-bank4.json),
versus 187.22/306.64 us for the prior interleaved dispatch entry and
310.01/411.28 us for the same-run CANN chain. All 30 samples, covering the
six execution orders five times, favor the selected entry. Every arm
includes independent shared input DynamicQuant; final addition, routed
scaling and TP reduction remain outside. Each bank independently passes
eight changing-input checks. The timed fixtures select 81/158 distinct
experts at M=12/32; the model's actual routing distribution is needed for
whole-round extrapolation. The adapter defaults to `build-pipeline-front`.

The [shared-eight-expert comparison](evidence/ascend-decode/megakernel-device-20261021/a3-162/routed-pipeline-front-shared8-summary.json)
repeats each fixture's first token's eight expert IDs across all rows,
retaining hidden inputs and slot probabilities. With 24 independent weight
banks and 24 calls, the selected entry takes 113.48/162.33 us at M=12/32,
versus 220.77/243.38 us for the same-run CANN chain. All 30 samples favor
the native entry, and each of the 24 banks passes eight changing-input
replays at all 37 boundaries. The active routed/shared weight payload is
270 MiB per arm, exceeding the platform configuration's 192 MiB L2 capacity.
This is a synthetic routing scenario; the earlier mixed-route measurements
use a different bank count.

A [Down-projection candidate](evidence/ascend-decode/megakernel-device-20261021/a3-162/routed-pipeline-front-striped-v2-summary.json)
splits each active expert's 6144 output columns across three AICs when
24 groups serve at most eight experts. Its K128 reduction remains unchanged.
It passes 96 changing replays across four widths and the 16/24-group routing
branches, plus 384 replays in the 24-bank timing fixtures. Initial M=12/32
medians are 116.63/159.92 us versus the same-run selected entry's
118.01/163.08 us. Median paired savings are 1.36/3.19 us, with 30 wins at
each width. Those initial measurements use separate native workspaces.
A [shared-workspace comparison](evidence/ascend-decode/megakernel-device-20261021/a3-162/routed-pipeline-front-striped-v2-placement-summary.json)
then aliases all 17 writable native buffers while preserving each library's
three byte-identical tiling objects. At M=12 the candidate loses 2.51/3.83 us
on the two arenas, with zero wins in 60 pairs; at M=32 it saves 2.35/2.15 us,
with 60 wins in 60 pairs. Each arm is replayed and copied separately after
timing to verify its outputs. DynamicQuant temporaries remain graph-specific.
At M=32 with 158 active experts, where the striped path is disabled, the
candidate loses 2.85/2.33 us on the two arenas, with zero wins in 60 pairs.
The hidden inputs and probabilities match the shared-eight comparison.
The candidate has not replaced the selected implementation; the M=32 gain
is limited to shared-eight routing.

An independent [gate/up K-split candidate](evidence/ascend-decode/megakernel-device-20261021/a3-162/routed-pipeline-front-splitk-v1-summary.json)
assigns three K2048 products per expert when 24 groups serve at most eight
active experts. It sums INT32 partials before the original FP16 boundary.
Across M=1,2,12,32, changing routes and the 16-group fallback, 104 graph
replays pass all 3,848 raw-byte boundary comparisons. Additional scratch is
384 KiB at M=12 and 1 MiB at M=32. The
[shared-workspace timing](evidence/ascend-decode/megakernel-device-20261021/a3-162/routed-pipeline-front-splitk-v1-placement-summary.json)
uses the same first INT32 plane for the control and candidate, retaining
the candidate's two additional planes. Shared-eight routing saves
1.39/1.48 us at M=12 and 0.27/1.46 us at M=32 across the two arenas.
Mixed routing with 158 active experts instead loses 5.27/6.20 us, with
zero wins in 60 pairs. All 1,256 numerical replays and 46,472 raw-byte
boundary comparisons pass. The candidate is not selected.

An independent [active-expert list candidate](evidence/ascend-decode/megakernel-device-20261021/a3-194/routed-pipeline-front-active-v1-summary.json)
stores the sorted nonempty expert IDs beside the original route metadata.
Both input entries produce this 1,056-byte table, and AIC/AIV consumers
traverse it while preserving projection calls, row ownership, arithmetic
and all six barriers. On a3-194, all 1,664 numerical graph replays pass
63,232 raw-byte comparisons across M=1,2,12,32, 16/24 groups, both BF16 and
dispatch entries, and transitions from 256 active experts back to eight.
The added boundary checks the complete table, including zeroed unused
entries. Eight 24-bank timing fixtures share all 17 common writable scratch
addresses and synchronization storage between the native arms, then repeat
with the other original allocation set. At M=12, paired median savings are
5.81/4.63 us with 81 routed experts and 22.40/19.72 us when every token selects
the same eight experts; all 120 pairs favor the candidate. At M=32, the
eight-expert fixture saves 16.56/19.65 us, but 158-expert routing loses
0.88/1.47 us in all 60 pairs. All 32 post-timing output checks pass. These
results support M=12 integration, with the parent retained for other widths;
M=1/2 have no timing evidence. The [actual adapter checks](evidence/ascend-decode/megakernel-device-20261021/a3-194/routed-front-active-adapter-v2-summary.json)
pass 96 graph replays and 3,568 raw-byte comparisons with mixed and changing
expert sets, FP32/BF16 shared scales, and 16/24 groups. They verify shape
selection and capture workspace ownership; the active entry runs only at
M=12 with 24 groups. Final addition and TP reduction are outside the measured
scope.

The selected parent also has a [callable device body](../tests/glm52/whole_layer/routed_body/README.md)
with caller-owned pipe and initialization. Its numerical statements and six
internal barriers are preserved. The standalone wrapper passes
[96 graph replays and 3,552 raw-byte comparisons](evidence/ascend-decode/megakernel-device-20261021/a3-194/routed-body-v2-summary.json)
against the parent. A containing kernel must explicitly publish and
synchronize its final stores before consuming them or reusing UB.

The [FFN bridge](../tests/glm52/whole_layer/ffn_bridge/README.md) provides
residual/norm, BF16 gate projection with FP32 logits, top-8 selection and
shared-expert input quantization. Its isolated
[fixed-logit routing gate](evidence/ascend-decode/megakernel-device-20261021/integration/ffn-bridge-topk-v2-a3-194/summary.json)
passes eight captures, 64 changing-input graph replays and 128 raw-byte
comparisons against the pinned official `MoeGatingTopK` operator. Expert IDs
and FP32 probabilities match at M=1,2,12,32 with 16/24 groups, including
tied scores, negative correction biases and extreme logits. This gate
exercises routing only. The production gate stores
weights as [256,6144] and passes a transpose to its FP32-output matrix
multiply, so its physical layout and runtime entry must be preserved in
that comparison.
The [actual gate-entry comparison](evidence/ascend-decode/megakernel-device-20261021/integration/ffn-gate-entry212-v2-a3-194/summary.json)
executes this call at M=1,12,32 with ND and NZ weights. All six calls return
finite FP32 outputs under Torch NPU 2.12. The fixed 2.10 runtime lacks the
`aten::mm.dtype` NPU implementation and all six calls fail in its CPU
fallback. Complete bridge comparisons use the 2.12 runtime with separately
rebound native and official-oracle libraries.
The [complete bridge comparison](evidence/ascend-decode/megakernel-device-20261021/integration/ffn-bridge212-v1-summary.json)
passes all eight M=1,2,12,32 and 16/24-group configurations with ND gate
weights. Its 64 changing-input graph replays pass 512 raw-byte comparisons:
normalized BF16 values, retained residual, rstd, FP32 logits, probabilities,
expert IDs, shared A8 codes and shared scales. Including the independent
TopK and epilogue paths, the same run set has 24 captures, 192 replays and
768 passing comparisons. These bridge runs exclude the routed/shared body
and TP reduction.
The separate [FFN output epilogue](evidence/ascend-decode/megakernel-device-20261021/integration/ffn-epilogue210-v1-a3-194/summary.json)
also passes eight captures, 64 changing-input replays and 128 raw-byte
comparisons at M=1,2,12,32 with 16/24 groups. It preserves the BF16 store
after multiplying the routed output by 2.5, then adds the shared output
with a second BF16 store. The cases include negative zero and alternating
magnitude-64 inputs.

The [complete local FFN](../tests/glm52/whole_layer/ffn_composed/README.md)
now executes the bridge, routed/shared body and epilogue in one mixed
AIC/AIV launch. Its [dual-reference comparison](evidence/ascend-decode/megakernel-device-20261021/integration/ffn-composed212-v2-summary.json)
passes all eight M=1,2,12,32 and 16/24-group configurations with NZ routed
weights and BF16 shared scales. Twenty-four graph captures cover 64
changing-input candidate replays and 128 reference replays; all 3,840
raw-byte comparisons across 30 intermediate and output boundaries pass.
One reference uses separate native launches; the other uses the pinned
official norm, gate, routing and quantization entries plus separate BF16
output arithmetic. Both retain the previously accepted native expert body.
The [matched-address timing](evidence/ascend-decode/megakernel-device-20261021/integration/ffn-composed-timing-v3-a3-194/summary.json)
at M12/24 with 65 active experts finds no launch-fusion gain. With 16 distinct
weight banks, the two allocation-owner arenas give fused/separate medians
of 236.917/235.734 us and 234.607/234.272 us. Paired median savings are
-1.155 and -0.337 us, with zero wins in each set of 30 pairs. All 19,200
numerical boundary checks and 360 event-tail checks pass. The pair uses
identical addresses for all 32 common scratch tensors; input preparation
and output poisoning occur outside timing. Attention composition and TP
reductions remain separate acceptance gates.

The [complete FFN with active-expert enumeration](evidence/ascend-decode/megakernel-device-20261021/integration/ffn-active212-m12-v1-a3-194/summary.json)
passes M12/24 with both references: 496 raw-byte comparisons cover the 30
existing boundaries and the entire INT32[264] active table, including padding.
Its [complete-FFN paired timing](evidence/ascend-decode/megakernel-device-20261021/integration/ffn-active-timing-v3-a3-194/summary.json)
with 65 active experts and 16 independent weight banks saves 5.659/9.549 us
in the two allocation-owner arenas, with 30/30 wins in each. All 19,200
numerical boundary checks, 360 event-tail checks and 348 full-table checks
pass. The comparison uses identical addresses for all 32 common scratch
tensors and requires the candidate to replace nonzero table sentinels.

The independent [body-argument descriptor](../tests/glm52/whole_layer/ffn_descriptor/README.md)
preserves the public ABI, all 32 scratch buffers and numerical statements.
Its [M12/24 paired timing](evidence/ascend-decode/megakernel-device-20261021/integration/ffn-descriptor-timing-v1-a3-194/summary.json)
with the same 65-expert fixture saves 5.192/2.423 us across the two owning
arenas, with all 60 pairs faster. All 19,200 numerical comparisons and
360 event-tail checks pass. This is a complete-FFN measurement; it does not
locate the gain within the body or measure the combined effect of descriptor
and active-expert enumeration.

The [combined active-list and descriptor candidate](../tests/glm52/whole_layer/ffn_active_descriptor/README.md)
passes 496 raw-byte checks at M12/24, including the entire active table.
Its [direct comparison against active FFN](evidence/ascend-decode/megakernel-device-20261021/integration/ffn-active-descriptor-timing-v1-a3-194/summary.json)
saves 19.813/19.851 us with 30/30 wins in each owner arena. Combined/active
medians are 224.177/243.945 us and 226.389/246.341 us. Both arms share all
33 tensor addresses and replace the same active-table sentinels. All
19,200 numerical checks, 360 event-tail checks and 696 full-table checks
pass. The result establishes a complete-FFN gain for this M12/24,
65-active-expert fixture; it does not identify the phase producing that gain.

The independent [token-owned output candidate](../tests/glm52/whole_layer/ffn_token_output/README.md)
keeps each token's route reconstruction, ordered weighted sum and final BF16
operations on the same AIV. All original diagnostic outputs and barriers
remain. Its [eight-geometry gate](evidence/ascend-decode/megakernel-device-20261021/integration/ffn-token-output212-wide-v1-a3-194/summary.json),
including the [M12/24 gate](evidence/ascend-decode/megakernel-device-20261021/integration/ffn-token-output212-m12-v1-a3-194/summary.json),
passes 3,840 raw-byte checks at M=1,2,12,32 and 16/24 groups.
The [M12/24 paired comparison](evidence/ascend-decode/megakernel-device-20261021/integration/ffn-token-output-timing-v1-a3-194/summary.json)
saves 2.125/5.506 us against the unmodified one-launch FFN in the two owner
arenas, with all 60 pairs faster. All 19,200 numerical and 360 event-tail
checks pass. This candidate does not include active-expert enumeration or
the descriptor change; their combined benefit requires direct measurement.

The [token-output tail-join candidate](../tests/glm52/whole_layer/ffn_token_join/README.md)
removes the now-empty body barrier and one adjacent outer join while retaining
the final publication, mixed-core join and resource reset. Its
[eight-geometry checks](evidence/ascend-decode/megakernel-device-20261021/integration/ffn-token-join212-wide-v1-a3-194/summary.json),
including the [M12/24 gate](evidence/ascend-decode/megakernel-device-20261021/integration/ffn-token-join212-m12-v1-a3-194/summary.json),
pass all 3,840 raw-byte comparisons. The
[direct comparison with its token-output parent](evidence/ascend-decode/megakernel-device-20261021/integration/ffn-token-join-timing-v1-a3-194/summary.json)
saves 11.445/7.748 us in the two owner arenas, with all 60 pairs faster.
All 19,200 numerical and 360 event-tail checks pass. These results use
M12/24, 65 active experts and 16 weight banks; other candidate combinations
require their own comparisons.

The [active-descriptor and token-tail composition](../tests/glm52/whole_layer/ffn_active_token_join/README.md)
passes all [3,968 raw-byte checks across eight geometries](evidence/ascend-decode/megakernel-device-20261021/integration/ffn-active-token-join212-wide-v1-a3-194/summary.json),
including [M12/24](evidence/ascend-decode/megakernel-device-20261021/integration/ffn-active-token-join212-m12-v1-a3-194/summary.json),
against both reference paths across eight changing inputs per geometry. It retains the
33-buffer interface and checks all 264 active-table words. The routed-body UB
allocation is 116,480 bytes, with no additional GM allocation. The preceding
norm allocates 195,584 bytes and resets its arena before the body. Its
[same-address comparison with the active-descriptor parent](evidence/ascend-decode/megakernel-device-20261021/integration/ffn-active-token-join-timing-v1-a3-194/summary.json)
saves 6.843/15.850 us, with all 60 pairs faster. Candidate/parent medians are
216.462/223.297 us and 211.493/227.337 us across the two owner arenas.
All 19,200 numerical, 360 event-tail and 696 full active-table checks pass.
These measurements use M12/24, 65 active experts and 16 weight banks.

The [terminal-completion variant](../tests/glm52/whole_layer/ffn_terminal/README.md)
retains all five body joins and changes only the final global-entry join to
local PIPE_ALL, DDR completion and ordinary pipe destruction. It passes
4,480 raw-byte checks across all eight configurations, including 512
immediate consumers in the captured graph. Its
[matched-address comparison](evidence/ascend-decode/megakernel-device-20261021/integration/ffn-terminal-timing-v1-a3-194/summary.json)
saves 8.834/8.019 us against the active-token-join parent, with all 60 pairs
faster and candidate medians of 211.257/211.925 us. All 19,200 numerical,
360 event-tail and 696 active-table checks pass. This completion rule applies
only to the global task boundary; the callable body's joins remain intact.

The independent [token-owned MSD split](evidence/ascend-decode/megakernel-device-20261021/integration/ffn-token-split212-wide-v2-a3-194/summary.json)
splits each A8 token once per 2048-column tile, then scatters the packed codes
to its eight original expert slots. All eight M1/2/12/32 by 16/24-block
geometries pass 3,968 raw-byte checks across eight changing inputs each,
including all active-expert-table words, against both existing references.
The 33-buffer interface and 116,480-byte routed-body UB allocation are unchanged.
Its [matched-address timing](evidence/ascend-decode/megakernel-device-20261021/integration/ffn-token-split-timing-v1-a3-194/summary.json)
saves 7.539/9.261 us against the active-token-join parent, with all 60 pairs
faster. Candidate/parent medians are 209.005/216.567 us and
208.502/217.774 us across the two owner arenas. All 19,200 numerical,
360 event-tail and 696 active-table checks pass. This uses the same M12/24,
65-active-expert, 16-bank scope.

The [combined split/terminal candidate](../tests/glm52/whole_layer/ffn_token_terminal/README.md)
passes all eight geometries with 4,480 raw-byte comparisons, including
512 immediate graph-consumer checks. Its
[matched-address timing](evidence/ascend-decode/megakernel-device-20261021/integration/ffn-token-terminal-timing-v1-a3-194/summary.json)
saves 6.195/13.502 us against the same active-token-join parent, with all 60
pairs faster. Candidate/parent medians are 211.532/217.734 us and
204.773/218.293 us across the two owner arenas. All 19,200 numerical,
360 event-tail and 696 active-table checks pass. The measured combined gain
depends on the address arena; the separate candidates' gains are not additive.

The independent [GU static-tiling candidate](../tests/glm52/whole_layer/ffn_gu_static/README.md)
makes the existing N/K and resource constants visible to the compiler while
preserving dynamic expert row counts and the original matrix operations.
Its [matched-address timing](evidence/ascend-decode/megakernel-device-20261021/integration/ffn-gu-static-timing-v1-a3-194/summary.json)
saves 5.434/6.167 us against the split/terminal parent at M12/24 and 65 active
experts, with all 60 pairs faster. Candidate/parent medians are
212.037/217.545 us and 208.779/214.910 us across two arenas. All 19,200 numerical,
360 event-tail and 696 active-table checks pass. Both arms retain the earlier
shared quantization. The corrected original-shared composition and its
independent numerical and timing comparisons are recorded above.

The independent [Down/route overlap candidate](../tests/glm52/whole_layer/ffn_down_overlap/README.md)
publishes each 256-column CUBE result to its two paired AIV consumers with
a four-credit mode-2 protocol. It retains the original full INT32 GM output,
all six body barriers and both outer joins. Its
[M12/24 dual-reference gate](evidence/ascend-decode/megakernel-device-20261021/integration/ffn-down-overlap212-m12-v1-a3-194/summary.json)
passes 480 raw-byte checks across eight changing inputs. Its
[same-address comparison](evidence/ascend-decode/megakernel-device-20261021/integration/ffn-down-overlap-timing-v1-a3-194/summary.json)
adds 99.411/101.369 us in the two owner arenas, with zero wins in 60 pairs;
all 19,200 numerical and 360 event-tail checks pass. The 128-column variant
is rejected. Its consumer slices increase route-helper calls twelvefold;
the measurement does not isolate the cost of those calls from scheduling
and synchronization.
The [four-tile batch](evidence/ascend-decode/megakernel-device-20261021/integration/ffn-down-batch-timing-v1-a3-194/summary.json)
reduces those helper calls fourfold and passes the same 19,200 numerical and
360 event-tail checks, but still adds 62.431/58.885 us against the original
FFN, with zero wins in 60 pairs. It is also rejected.

The separate
[phase profile](evidence/ascend-decode/megakernel-device-20261021/integration/ffn-profile212-m12-v1-a3-194/summary.json)
also passes all 480 numerical comparisons and validates 13 clock marks on
each of 72 workers. These worker-local windows include waits and
instrumentation. Its [16-bank event comparison](evidence/ascend-decode/megakernel-device-20261021/integration/ffn-profile-timing-v3-a3-194/summary.json)
measures 12.961/13.306 us of added latency in the two arenas, with all 60
pairs slower. This includes final trace stores outside the local windows;
the measured overhead cannot be assigned wholesale to any one phase.
The [two-clock variant](evidence/ascend-decode/megakernel-device-20261021/integration/ffn-profile-sparse212-m12-v1-a3-194/summary.json)
preserves the logical trace-store footprint and samples only body setup.
It passes the same 480 numerical comparisons and reports 14.88–14.98 us
maximum worker-local setup windows across eight distinct inputs. Its
[matched event comparison](evidence/ascend-decode/megakernel-device-20261021/integration/ffn-profile-sparse-timing-v1-a3-194/summary.json)
still adds 11.079/10.547 us, with zero wins in all 60 pairs and all numerical
and sparse-trace checks passing. This bounds the diagnostic's total cost;
an uninstrumented candidate comparison is needed to establish a setup gain.

At the historical 5.441 accepted tokens per round, 2 ms TPOT permits
10.882 ms per round, or at most 139.51 us per layer across 78 layers before
any work outside those layers. The complete M12 FFN fixture with 65 active
experts takes 200.96–202.56 us across the corrected static-GU variant's
two arenas. Applying that fixture to all 75 MoE layers would take
15.07–15.19 ms before
attention, TP reductions, dense layers and other work. This is a budget illustration
using one synthetic route distribution. A 2 ms engine result still needs
a shorter critical path and a whole-model measurement with actual routes
and accepted-token counts.

The routed stage holds all 256 TP-sharded experts on each rank, with intermediate
width 128. It preserves each token's original top-8 slot, rounds each expert
output to BF16, and combines slots in order using FP32. Weights are copied from
ModelSlim ND storage before vendor NZ conversion by default. The optional
`TILERT_ROUTED_WEIGHT_LAYOUT=nz` path instead borrows final vendor INT32 NZ
carriers and FP32 bias tensors, retaining only FP32 scale copies. This removes
21.09375 GiB of duplicate packed weights per rank across 75 MoE layers and
keeps the vendor prefill path. The NZ-capable library has compiled separately
in `build-nz` and passed numerical and changing-route replay comparisons against
the ND CANN chain at M=1,2,12,32. Include the selected path's actual allocations
in HBM admission.

BF16 inputs retain vendor `MoeInitRouting` dispatch quantization. The adapter
recovers each original token's INT8 row and scale from its slot-zero inverse
index before native execution. Already-quantized inputs reuse their scales.
Dispatch, recovery and route-metadata work remain outside the native launch
and must be included when measuring the replacement path.

The optional `TILERT_ROUTED_REUSE_DISPATCH=1` path consumes the dispatch
operator's sorted rows, inverse permutation and expert counts directly.
It reconstructs original token rows and route metadata inside the native
launch, retaining original-slot probability rounding and combine order.
Use its separate `build-route` library and include the external dispatch
operator in the measured path.

The MLA stage consumes BF16 `[M,4,512]` latent outputs and returns the local
BF16 `[M,6144]` projection. Its adapter invokes the existing TP reduction.
With `--mla-tail` alone, attention prolog, KV/index-cache updates, sparse
selection, SFA, residuals, normalization, routing selection, shared experts
and collectives retain their existing implementations. The first three dense
layers also remain required.

The native adapters own persistent scratch and prepare it before graph capture.
Layers share scratch on the same stream; independent streams use separate
storage. The routed stage currently reserves 35,030,528 bytes at M=32 and
256 experts, excluding weights; the dispatch-metadata entry uses 35,232,384 bytes.
Captured and eager arenas are distinct. Native
outputs retained beyond the next call need an explicit copy.
Direct launches obtain the current stream through `stream()`, which submits
queued Torch producers before the AscendC runtime call. `stream(false)` can
launch before pending dispatch or gather operations and read stale inputs.

## Build and connect

Use an A3 CANN installation with its matching torch/torch_npu environment.
Both build scripts default to a `build/` directory beside their sources.

```bash
CANN_ROOT=/path/to/cann PYTHON_BIN=/path/to/python \
  bash tests/glm52/routed_int4/build_so.sh
CANN_ROOT=/path/to/cann PYTHON_BIN=/path/to/python \
  bash tests/glm52/mla_tail/build_so.sh
```

The MLA adapter requires an explicit hook in the vLLM-Ascend attention backend.
Apply [the hook patch](../patches/vllm-ascend/mla-tail-hook.patch) to the checkout
used by the engine after checking that it applies cleanly. Its default method
retains the original value-up, optional gate and output-projection order. The
adapter requires an ungated dynamic projection and keeps the original method
for prefill, empty local queries and context-parallel gathering.

`CannModel(..., routed_kernel=True, mla_tail=True)` enables both stages.
`tests/glm52/bench_cann_mtp.py` exposes the corresponding command-line flags.
`TILERT_ROUTED_LIB_DIR` and `TILERT_MLA_TAIL_LIB_DIR` override library locations.
The routed kernel requires EP off and is mutually exclusive with `--narrow-bank`.

`--routed-shared-kernel` also installs the routed adapter, selecting borrowed
NZ weights and dispatch reuse. Apply the
[routed callback patch](../patches/vllm-ascend/routed-shared-hook.patch) to the
engine's vLLM-Ascend checkout. The default callback remains its quant method;
the replacement runs within the original prepare/top-k/finalize and context
lifetimes. The owning runner retains routed scaling, output addition and TP
reduction. Supported calls use TP16, EP1, DP1, PCP1, no sequence parallelism or
routed transforms, and an ungated, unclipped W8A8 shared expert. Prefill above
32 rows and active LoRA keep the original runner path. Original shared scales
must be FP32 or BF16; both types have passed device comparisons. The installed
CANN NZ QMM rejects FP16 scales before execution.

The routed/shared adapter selects the active-expert table at M=12 with
24 core groups and the parent library at other shapes. Workspace preparation
uses the same choice before capture; both library hashes enter the model
fingerprint. Build the active variant from the selected parent's frozen
source manifest, then set the three library paths before starting:

```bash
bash tests/glm52/routed_int4/build_balanced_promoted.sh
routed_snapshot=docs/evidence/ascend-decode/megakernel-device-20261021/a3-162/routed-pipeline-front-build
bash "$routed_snapshot/source/routed_int4/build_pipeline_front.sh" \
  "$PWD/tests/glm52/routed_int4/build-pipeline-front"
bash tests/glm52/routed_int4/build_pipeline_front_active.sh
TILERT_ROUTED_LIB_DIR="$PWD/tests/glm52/routed_int4/build-balanced-promoted" \
TILERT_ROUTED_SHARED_LIB_DIR="$PWD/tests/glm52/routed_int4/build-pipeline-front" \
TILERT_ROUTED_SHARED_ACTIVE_LIB_DIR="$PWD/tests/glm52/routed_int4/build-pipeline-front-active" \
  python tests/glm52/bench_cann_mtp.py --help
```

Use these library variables with the ordinary TP16 launcher and
`--routed-shared-kernel`. The adapter adds 2.25 MiB of shared INT8 tiles per MoE
layer, or 168.75 MiB across 75 layers, while retaining vendor weights for
prefill. Include scales, stream-specific pools and all captured widths in
HBM admission. The hook patch and CPU contract checks do not establish a
whole-model device result.

`--native-attention` installs the indexed and shared-indexer entries directly
on loaded backbone attention implementations. It requires SFA, is mutually
exclusive with `--mla-tail`, and preserves MTP and unsupported per-call paths.
The supported contract is TP16, one request, exact M=1–32, Prolog V3, BF16 main
caches, ungated dynamic-INT8 projections, and no context/sequence parallelism,
LoRA, KV transfer or weight reload. The model owns persistent top-k storage
and the original TP reduction; each distinct live slot source owns its INT64
conversion buffer. See the [adapter contract](../tests/glm52/whole_layer/attention_indexed.md).

```bash
OPS_TRANSFORMER_ROOT=/path/to/recorded/ops-transformer \
  bash tests/glm52/whole_layer/attention_indexed_build.sh
OPS_TRANSFORMER_ROOT=/path/to/recorded/ops-transformer \
  bash tests/glm52/whole_layer/attention_full_build.sh
```

`TILERT_INDEXED_ATTENTION_LIBRARY` and `TILERT_SHARED_ATTENTION_LIBRARY`
select the actual libraries. Set the latter to the newly built
`attention_full_build/whole_layer_full_attention.so`; the adapter's default
points to the measured `attention_full_build-dequant` artifact. Benchmark
records include enabled native flags, adapter/vendor source hashes and both
loaded library hashes. Include retained packed weights and every exact-width,
stream-specific pool in HBM admission.

## Verification

CPU checks exercise actual packing, route ownership, capture reservations and
shared C++ storage layouts:

```bash
python -m pytest -q tests/glm52/test_routed_adapter.py \
  tests/glm52/test_routed_shared_adapter.py \
  tests/glm52/test_attention_native_adapter.py \
  tests/glm52/test_mla_tail_adapter.py tests/glm52/test_fused_workspaces.py
python -m unittest discover -s tests/glm52/mla_tail -v
c++ -std=c++17 -O2 -Wall -Wextra -Werror \
  tests/glm52/routed_int4/contract_test.cpp -o /tmp/routed-contract
/tmp/routed-contract
```

The native stage READMEs provide device differential and replay commands.
Compare with the deployed CANN chain on identical weights and activations,
including zero rows, signed values, hot experts, changing routes and every
captured width. Preserve raw errors and intermediate boundaries before timing.
The two stages have different activation quantization contracts: fused GMM1
uses FP16 before integer rounding; the MLA tail follows standalone DynamicQuant.

After device correctness and changing-input graph replay pass, run the existing
[whole-engine arm launcher](../tests/glm52/launch_arm.sh) with alternating control
and candidate arms. Keep checkpoint, prompts, drafter, width, cache, TP settings
and token-delivery timing scope fixed. Require successful non-warmup requests,
identical emitted tokens, and the full callback TPOT result. A local operator
speedup establishes neither a whole-layer saving nor the 2 ms target.

The [ROCm whole-layer interface](../tilert/models/glm_5_2_rocm/ops/pure_mla_moe_layer.py)
provides the same-model orchestration reference. Numerical and execution
contracts are documented in [MSD INT4](reference/w4a8-msd-int4-mmad.md),
[MLA fusion](reference/mla-decode-fusion.md), and
[persistent decode](reference/megakernel-persistent-decode.md).
