# The layer costs 210 microseconds, and the 374 us gap was an artefact

A kernel census of `profile-rope-20261018`, rank 1, non-collective kernels:

    690 distinct kernels, 5544 calls, 179.14 ms

    AivKernel                                   64   54.94 ms   858.36 us
    sk_1277_full_model_start_aiv_all_reduce      3    9.85 ms  3282.35 us
    aclnnFusedInferAttentionScoreV4            244    9.14 ms    37.46 us
    aclnnInplaceCopy_CastAiCore_Cast          1590    4.31 ms     2.71 us
    sk_681_full_model_start_static               3    0.65 ms   215.38 us
    sk_1195_full_model_start_static              3    0.64 ms   212.67 us
    sk_1096_full_model_start_static              3    0.64 ms   211.75 us
    sk_799_full_model_start_static               3    0.63 ms   211.24 us
    ... dozens more of the same shape, 209 to 215 us each

## What the names say

`sk_*_full_model_start_static` is the SuperKernel: one per layer, called three
times -- once a measured request -- at **209 to 215 us**. That is the layer,
and it matches the synthetic layer's in-graph 214 us to within two percent.

So a layer costs about 210 us in the engine, not 588.

## Which makes the 374 us gap an artefact of my own arithmetic

The 588 us came from dividing the round model's fitted fixed term, 45.88 ms, by
78 layers. That fit holds everything in a round -- the standalone collectives,
the drafting, the Gloo command, the MTP recomputation, prefill remnants in a
six-token capture -- and attributing all of it to layers put 374 us a layer
where no layer spends it.

Four explanations were measured and excluded for a quantity that did not exist:
host-side issue, the native device cost, arrival skew, and message size. Each
exclusion is sound on its own terms and none of them was the answer, because
the question was wrong.

## The round's actual shape, from this census

    78 SuperKernels at 210 us                16.4 ms
    64 standalone collectives at 858 us      54.9 ms over 3 requests, 18.3 a request
    244 attention calls at 37.46 us           9.1 ms over 3, 3.0 a request
    1590 casts at 2.71 us                     4.3 ms over 3, 1.4 a request

The collectives dominate and they are outside the SuperKernel. That is where
the round's time goes, and the capture's six-token scope means the absolute
figures still need a steady-state run -- but the structure is now right.
