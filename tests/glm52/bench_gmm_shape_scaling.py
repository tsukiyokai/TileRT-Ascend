"""Time the W4A8 grouped matmul at one active-expert count and row count.

Each active group reads its own weight shard, so sweeping the number of
non-zero groups scales HBM traffic at a fixed kernel shape and separates the
call's fixed cost from its streaming slope; sweeping rows per group scales
the activation side instead. Values do not affect timing, so the bank is
filled with arbitrary bits. One shape per process. One device, one rank.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter_ns

import torch

EXPERTS = 256
HIDDEN = 6144
LOCAL_INTERMEDIATE = 128
PACK = 8
PROJECTIONS = {"gate_up": (HIDDEN, 2 * LOCAL_INTERMEDIATE),
               "down": (LOCAL_INTERMEDIATE, HIDDEN)}


def build_bank(inputs, outputs, device):
    return {
        "weight": torch.randint(-2 ** 31, 2 ** 31 - 1,
                                (EXPERTS, inputs, outputs // PACK),
                                dtype=torch.int32, device=device),
        "scale": torch.ones(EXPERTS, 1, outputs, dtype=torch.int64,
                            device=device),
        "scale_bias": torch.zeros(EXPERTS, outputs, dtype=torch.float32,
                                  device=device),
    }


def build_activation(rows, inputs, device):
    return {
        "x": torch.ones(rows, inputs, dtype=torch.int8, device=device),
        "per_token_scale": torch.ones(rows, dtype=torch.float32,
                                      device=device),
    }


def grouped_matmul_call(bank, activation, counts):
    import torch_npu

    def call():
        return torch_npu.npu_grouped_matmul(
            x=[activation["x"]], weight=[bank["weight"]],
            scale=[bank["scale"]], bias=[bank["scale_bias"]],
            per_token_scale=[activation["per_token_scale"]],
            group_list=counts, split_item=2, group_type=0,
            group_list_type=1, output_dtype=torch.bfloat16,
        )[0]

    return call


def batch_mean_us(call, calls):
    torch.npu.synchronize()
    start = perf_counter_ns()
    for _ in range(calls):
        call()
    torch.npu.synchronize()
    return (perf_counter_ns() - start) / calls / 1e3


def measure(args, report):
    import torch_npu

    torch_npu.npu.set_device(args.device)
    device = torch.device("npu", args.device)
    inputs, outputs = PROJECTIONS[args.projection]
    bank = build_bank(inputs, outputs, device)
    activation = build_activation(args.experts * args.rows, inputs, device)
    counts = torch.zeros(EXPERTS, dtype=torch.int64, device=device)
    counts[:args.experts] = args.rows
    call = grouped_matmul_call(bank, activation, counts)

    for _ in range(args.warmup):
        call()
    samples = sorted(batch_mean_us(call, args.calls)
                     for _ in range(args.batches))

    median = statistics.median(samples)
    read = args.experts * inputs * (outputs // PACK) * 4
    report.update({
        "status": "pass", "weight_bytes": read,
        "median_us": median, "min_us": samples[0],
        "p90_us": samples[-1 - len(samples) // 10],
        "gb_per_s": read / (median / 1e6) / 1e9,
    })
    return True


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--projection", choices=tuple(PROJECTIONS),
                        default="gate_up")
    parser.add_argument("--experts", type=int, required=True)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--batches", type=int, default=20)
    return parser.parse_args()


def main():
    args = arguments()
    report = {
        "status": "error", "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv], "hostname": platform.node(),
        "projection": args.projection, "experts": args.experts,
        "rows": args.rows, "device": args.device, "calls": args.calls,
        "warmup": args.warmup, "batches": args.batches,
        "scope": "one die, one rank, shared with another tenant",
        "value_note": "arbitrary bits; timing does not depend on values",
        "bank_note": "the full 256-group bank is resident; counts select the "
                     "active groups",
    }
    try:
        passed = measure(args, report)
    except Exception as error:
        report["error"] = repr(error)[:2000]
        passed = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"projection": args.projection, "experts": args.experts,
                      "rows": args.rows, "passed": passed,
                      "median_us": report.get("median_us"),
                      "gb_per_s": report.get("gb_per_s")}))
    return int(not passed)


if __name__ == "__main__":
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    raise SystemExit(main())
