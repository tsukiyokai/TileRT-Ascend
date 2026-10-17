"""Plot frozen CANN/HF hidden-state differences without an acceptance gate."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys


ROUND_OFF = 2 ** -8


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def collect(native, reference):
    import torch

    report_path = native / "rank-0.json"
    manifest_path = reference / "manifest.json"
    report = json.loads(report_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if (report["checkpoint_sha256"] != manifest["metadata_sha256"]
            or report["inputs"]["prefix_sha256"]
            != manifest["prefix_files_sha256"]["rank-0.pt"]):
        raise ValueError("CANN and HF must use the same checkpoint and prefix")
    artifacts = {str(path): digest(path)
                 for path in (report_path, manifest_path)}
    rows = []
    for step in range(report["steps"]):
        position = report["initial_position"] + step
        name = f"step-{position:05d}.pt"
        traces = []
        for directory, hashes in (
            (native / "rank-0", report["trace_sha256"]),
            (reference, manifest["trace_sha256"]),
        ):
            path = directory / name
            artifacts[str(path)] = digest(path)
            if artifacts[str(path)] != hashes[name]:
                raise ValueError(f"Frozen trace hash differs: {path}")
            traces.append(torch.load(path, map_location="cpu",
                                     weights_only=True))
        actual, expected = traces
        if (actual["token"] != expected["token"]
                or actual["position"] != expected["position"]
                or len(actual["hidden_states"]) != 79
                or len(expected["hidden_states"]) != 79):
            raise ValueError("Expected matching inputs and 79 hidden states")
        for layer, (left, right) in enumerate(zip(
            actual["hidden_states"][1:], expected["hidden_states"][1:],
        )):
            if (left.shape != right.shape or left.shape != (1, 6144)
                    or left.dtype != right.dtype
                    or left.dtype != torch.bfloat16):
                raise ValueError("Expected matching BF16 [1,6144] states")
            left, right = left.double(), right.double()
            delta = left - right
            nrmse = float(delta.norm() / right.norm())
            recorded = next(c["nrmse"] for c in report["checks"]
                            if c["name"] ==
                            f"hf.{step}.hidden_states.{layer + 1}")
            rows.append({
                "decode_position": position, "consumed_token": actual["token"],
                "layer": layer, "hidden_state_index": layer + 1,
                "boundary": ("merged_residual" if layer < 77
                             else "layer77_and_final_rmsnorm"),
                "nrmse": nrmse, "nrmse_over_u": nrmse / ROUND_OFF,
                "recorded_fp32_nrmse": recorded,
                "delta_rms": float(delta.square().mean().sqrt()),
                "reference_rms": float(right.square().mean().sqrt()),
                "mean_signed_error": float(delta.mean()),
            })
    return rows, artifacts, report


def draw(rows, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False})
    figure, axis = plt.subplots(figsize=(11, 5.5), layout="constrained")
    positions = sorted({row["decode_position"] for row in rows})
    colors = ("#2563eb", "#d65f18")
    for position, color in zip(positions, colors):
        series = [r for r in rows if r["decode_position"] == position]
        raw, final = series[:-1], series[-1]
        label = f"Position {position}, token {series[0]['consumed_token']}"
        axis.plot([r["layer"] for r in raw],
                  [100 * r["nrmse"] for r in raw], color=color,
                  linewidth=1.8, label=label)
        axis.scatter([77], [100 * final["nrmse"]], s=55, marker="D",
                     facecolors="white", edgecolors=color, zorder=4)
        peak = max(raw, key=lambda r: r["nrmse"])
        axis.annotate(f"L{peak['layer']}: {100 * peak['nrmse']:.2f}%",
                      (peak["layer"], 100 * peak["nrmse"]),
                      xytext=(8, 7), textcoords="offset points", color=color)
    axis.axhline(100 * ROUND_OFF, color="#64748b", linestyle="--",
                 linewidth=1, label="BF16 u = 2^-8 (relative scale)")
    axis.set(xlabel="Decoder layer (zero based)",
             ylabel="Hidden-state NRMSE (%)",
             xlim=(-0.5, 78.5), ylim=(0, None))
    axis.set_xticks([0, *range(8, 77, 8), 77])
    axis.grid(alpha=0.2)
    second = axis.secondary_yaxis(
        "right", functions=(lambda x: x / (100 * ROUND_OFF),
                            lambda x: x * 100 * ROUND_OFF),
    )
    second.set_ylabel("NRMSE / u (not elementwise ULP counts)")
    axis.legend(loc="upper left", frameon=False)
    figure.suptitle("GLM-5.2 W4A8C8 | A3 TP16 | batch 1 | 8K prefix",
                    fontsize=13)
    axis.set_title("Cross-configuration: CANN compressed MLA / HF expanded KV",
                   fontsize=10)
    figure.supxlabel(
        "Layers 0-76: merged residuals. "
        "Diamonds: layer 77 followed by final RMSNorm.",
        fontsize=9,
    )
    figure.savefig(output / "hidden-nrmse.png", dpi=180)
    figure.savefig(output / "hidden-nrmse.svg", metadata={"Date": None})
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    import torch

    torch.set_num_threads(2)
    rows, artifacts, native = collect(args.native_run, args.reference)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "hidden-nrmse.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    draw(rows, args.output)
    result = {
        "scope": "78 decoder boundaries, two frozen independent continuations",
        "comparison_kind": "cross_configuration", "representative_rank": 0,
        "formula": "norm(CANN - HF, 2) / norm(HF, 2); FP64 evaluation",
        "unit_roundoff": ROUND_OFF, "elementwise_ulp_count": False,
        "thresholds_changed": False, "performance_measured": False,
        "checkpoint_sha256": native["checkpoint_sha256"],
        "inputs": native["inputs"], "artifact_sha256": artifacts,
        "source": {"path": str(Path(__file__).resolve()),
                   "sha256": digest(__file__)},
        "command": [sys.executable, *sys.argv], "rows": rows,
    }
    assert not (hasattr(torch, "npu") and torch.npu.is_initialized())
    for name in ("hidden-nrmse.csv", "hidden-nrmse.png", "hidden-nrmse.svg"):
        result["artifact_sha256"][str(args.output / name)] = digest(
            args.output / name,
        )
    (args.output / "hidden-nrmse.json").write_text(
        json.dumps(result, indent=2) + "\n",
    )
    print(json.dumps({"rows": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
