"""Byte budget for the fixed TP16 CANN representation, without device reads."""

import argparse
import hashlib
import json
from pathlib import Path


HIDDEN = 6144
LAYERS = 78
MOE_LAYERS = 75
LOCAL_INTERMEDIATE = 128
VOCAB_SHARD = 9680


def w8(outputs, inputs):
    # Runtime keeps INT8 values and FP32 channel scales; offsets are checked
    # once on load, then discarded by NpuQuantizedLinear.
    return outputs * inputs + outputs * 4


def w4(outputs, inputs):
    # INT32 nibble packing preserves four bits per value. Scale carriers
    # are INT64 and MSD compensation is FP32 per output on this TP rank.
    return outputs * inputs // 2 + outputs * (8 + 4)


def ffn(intermediate, quantized):
    operation = w4 if quantized == 4 else w8
    return (operation(2 * intermediate, HIDDEN)
            + operation(HIDDEN, intermediate))


def components():
    attention = (
        w8(2048, HIDDEN) + w8(576, HIDDEN) + w8(1024, 2048)
        + w8(HIDDEN, 1024) + 4 * 448 * 512 * 2
        + (HIDDEN * 2 + 2048 + 512) * 2
    )
    indexer = (w8(4096, 2048) + 128 * HIDDEN * 2
               + 32 * HIDDEN * 2 + 2 * 128 * 4)
    router = 256 * HIDDEN * 4 + 256 * 4
    return {
        "attention": attention, "indexer": indexer, "router": router,
        "shared_ffn": ffn(128, 8), "dense_ffn": ffn(768, 8),
        "one_expert": ffn(LOCAL_INTERMEDIATE, 4),
        "head": VOCAB_SHARD * HIDDEN * 2,
        "embedding_row": HIDDEN * 2, "final_norm": HIDDEN * 2,
    }


def main_weights(experts_per_layer=8, token_rows=1):
    parts = components()
    return {
        "attention": LAYERS * parts["attention"],
        "indexer": 21 * parts["indexer"],
        "router": MOE_LAYERS * parts["router"],
        "shared_ffn": MOE_LAYERS * parts["shared_ffn"],
        "dense_ffn": 3 * parts["dense_ffn"],
        "experts": MOE_LAYERS * experts_per_layer * parts["one_expert"],
        "head": parts["head"], "final_norm": parts["final_norm"],
        "embedding_rows": token_rows * parts["embedding_row"],
    }


def draft_weights(token_rows=1):
    parts = components()
    return {
        "attention": parts["attention"], "indexer": parts["indexer"],
        "router": parts["router"], "shared_ffn": parts["shared_ffn"],
        "experts": 8 * parts["one_expert"], "head": parts["head"],
        "embedding_rows": token_rows * parts["embedding_row"],
        "rot": HIDDEN * HIDDEN * 2,
        "eh_proj": HIDDEN * (HIDDEN * 2) * 2,
        "enorm_hnorm_head_norm": HIDDEN * 3 * 2,
    }


def estimate():
    main_kv = LAYERS * 2048 * (512 + 64) * 2
    main_index = 8192 * (16 * 130 + 5 * 256)
    draft_kv = 2048 * (512 + 64) * 2 + 8192 * 256
    one = main_weights()
    one_bytes = sum(one.values()) + main_kv + main_index
    draft = draft_weights()
    one_draft_bytes = sum(draft.values()) + draft_kv
    skipped_index_query = w8(4096, 2048) + 32 * HIDDEN * 2
    one_chain_bytes = one_draft_bytes - skipped_index_query - 8192 * 256
    scenarios = []
    for experts in (8, 16, 32):
        for cache_passes in (1, 4):
            # One target-conditioned MTP4 forward plus two chained MTP1
            # forwards per steady M4 round. Initial seeding is separate
            # here; whole-request TPOT includes it after the first callback.
            # The target-conditioned forward may read 8..32 MTP experts.
            conditioned_draft = (
                sum(draft_weights(4).values()) + draft_kv
                + (experts - 8) * components()["one_expert"]
            )
            traffic = (
                sum(main_weights(experts, 4).values())
                + cache_passes * (main_kv + main_index)
                + 2 * one_chain_bytes + conditioned_draft
                + (cache_passes - 1) * draft_kv
            )
            for efficiency in (1.0, 0.7):
                hbm_ms = traffic / (1.6e9 * efficiency)
                # 162 block reductions plus four embedding reductions.
                # Greedy selection uses four additional all-gathers.
                ar_ms = 166 * 0.012
                scenarios.append({
                    "verify_width": 4, "expert_union_per_layer": experts,
                    "kv_read_passes": cache_passes,
                    "bandwidth_efficiency": efficiency,
                    "round_bytes": traffic, "hbm_round_ms": hbm_ms,
                    "allreduce_budget_ms": ar_ms,
                    "allreduce_calls": 166,
                    "allreduce_48kib_calls": 160,
                    "allreduce_12kib_calls": 6,
                    "greedy_allgather_calls": 4,
                    "accepted_for_2ms_hbm_only": hbm_ms / 2,
                    "accepted_for_2ms_serial_budget":
                    (hbm_ms + ar_ms) / 2,
                    "tpot_at_accept_3_2_serial_ms": (hbm_ms + ar_ms) / 3.2,
                    "tpot_at_accept_3_2_full_overlap_ms":
                    max(hbm_ms, ar_ms) / 3.2,
                })
    root = Path(__file__).resolve().parents[2]
    sources = [Path(__file__), *(root / "tilert/models/glm_5_2_ascend" / name
               for name in ("native_engine.py", "ops/quantized.py",
                            "ops/expert.py", "weights.py"))]
    return {
        "scope": "Analytical streaming-read budget; not a device measurement",
        "layout": "TP16, W4 INT64 scales, FP32 replicated router, BF16 head",
        "assumptions": [
            "Each weight is read once per participating expert group",
            "No L2 credit; no workspace, activation traffic or MSD scratch",
            "KV reads bracket one shared pass and four independent passes",
            "M4 main and MTP verify use the same expert-union scenario",
            "MTP rot and eh_proj are replicated in the current implementation",
            "AllReduce alpha=12us is a budget parameter, not a local result",
            "Greedy all-gather latency remains outside the AR budget",
            "Excludes Python, launch, host validation and output delivery",
            "Steady rounds exclude the initial seed; request TPOT includes it",
        ],
        "main_m1_weights_by_component": one,
        "main_m1_weight_bytes": sum(one.values()),
        "main_m1_kv_bytes": main_kv + main_index,
        "main_m1_read_bytes": one_bytes,
        "main_m1_hbm_peak_ms": one_bytes / 1.6e9,
        "main_m1_hbm_70pct_ms": one_bytes / (1.6e9 * 0.7),
        "main_m1_tok_s_peak_hbm": 1.6e12 / one_bytes,
        "draft_m1_weights_by_component": draft,
        "draft_m1_read_bytes": one_draft_bytes,
        "draft_chain_read_bytes": one_chain_bytes,
        "initial_seed_read_bytes": one_draft_bytes,
        "m1_collectives": {
            "block_allreduce_12kib": 156,
            "embedding_allreduce_12kib": 1,
            "greedy_allgather_per_rank_bytes": 8,
        },
        "m4_greedy_allgathers_per_rank_bytes": [32, 8, 8, 32],
        "m4_scenarios": scenarios,
        "source_sha256": {str(path.relative_to(root)):
                          hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = estimate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "main_m1_read_bytes", "main_m1_hbm_peak_ms",
        "main_m1_hbm_70pct_ms", "main_m1_tok_s_peak_hbm",
        "draft_m1_read_bytes",
    )}))


if __name__ == "__main__":
    main()
