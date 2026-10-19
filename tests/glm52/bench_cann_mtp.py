"""Measure TP16 M4 session delivery from a captured full 8K prefix.

The first output is prefill's g0. TPOT spans its callback through the last
callback, including draft, target verification, MTP recomputation, acceptance
and Gloo commands. Prefix readiness and reset are outside that window.
"""

import argparse
import ast
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter_ns
import traceback

from bench_cann_decode import bind_cpu_threads
from run_cann_mtp_rounds import load_inputs, source_hashes
from run_cann_model import digest

from tilert.models.glm_5_2_ascend.session import PrefixRequest
from tilert.pd_vllm.ascend_cache import (
    CannRankEngine, FullRankPrefix, checkpoint_fingerprint,
)
from tilert.pd_vllm.decode_timing import (
    TimedBackend, atomic_json, token_metrics,
)


ROOT = Path(__file__).resolve().parents[2]
PREFIX_LENGTH = 8192
WIDTH = 4


class MeasuredRank(CannRankEngine):
    """Retain local round states and finish prefix copies before delivery."""

    def __init__(self, target, draft, group):
        super().__init__(target, draft)
        self.group = group
        self.requests = []
        self.current = None

    def inject(self, prefix):
        import torch
        import torch.distributed as distributed

        super().inject(prefix)
        torch.npu.synchronize(self.target.device)
        distributed.barrier(group=self.group)
        self.current = {
            "injected": [self.position, self.draft.position],
            "first_token": self.first_token_id, "rounds": [],
        }
        self.requests.append(self.current)

    def advance_mtp(self):
        start = perf_counter_ns()
        result = super().advance_mtp()
        end = perf_counter_ns()
        commit = result.commit
        self.current["rounds"].append({
            "start_ns": start, "end_ns": end, "elapsed_ns": end - start,
            "width": commit.width, "candidates": list(result.candidates),
            "predictions": list(result.target_predictions),
            "accepted_tokens": list(commit.accepted_tokens),
            "natural_accepted": commit.natural_accepted,
            "matched_drafts": commit.matched_drafts,
            "before": commit.main.before, "written": commit.main.written,
            "retained": commit.main.retained,
            "draft_retained": commit.draft.retained,
        })
        return result

    def finish_mtp(self, retained_position):
        super().finish_mtp(retained_position)
        self.current["finished"] = [self.position, self.draft.position]

    def reset(self):
        super().reset()
        if self.current is not None:
            self.current["reset"] = [self.position, self.draft.position]
            self.current = None


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "capture-case", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--width", type=int,
                        choices=(2, 4, 6, 8, 12, 16, 24, 32),
                        default=WIDTH)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--capacity", type=int, default=9344)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--target-graph", action="store_true")
    parser.add_argument("--draft-graph", action="store_true",
                        help="Graph shared-index M1 draft chains")
    parser.add_argument("--conditioned-graph", action="store_true",
                        help="Graph the M4 target-conditioned MTP pass")
    parser.add_argument("--draft-projection-tp", action="store_true",
                        help="Shard MTP rot and eh_proj outputs over TP16")
    parser.add_argument("--hoist-metadata", action="store_true",
                        help="Build SFA/indexer metadata once per captured"
                             " width and refresh only its rope slices")
    parser.add_argument("--hoist-forward-context", action="store_true",
                        help="Serve a prebuilt MC2 mask per token width")
    parser.add_argument("--compile-target", action="store_true")
    parser.add_argument("--super-kernel", action="store_true",
                        help="Use static kernels and SuperKernel for target")
    parser.add_argument("--attention-kernel", choices=("sfa", "gather_fia"),
                        default="sfa")
    parser.add_argument("--fp32-head", action="store_true")
    parser.add_argument("--index-topk", type=int,
                        help="override the checkpoint's 2048 selected "
                             "positions; sparse attention costs 17.61 us "
                             "a thousand picks in the graph")
    parser.add_argument("--cpu-affinity", action="store_true")
    parser.add_argument("--rounds-per-command", type=int, default=1,
                        help="Rounds each broadcast command batches")
    parser.add_argument("--expert-parallel", action="store_true",
                        help="EP16 experts and the fused MC2 MoE operator")
    parser.add_argument("--onesided-reduce",
                        help="Config-store endpoint, e.g. tcp://127.0.0.1:9421,"
                             " replacing the TP sum with a one-sided push")
    args = parser.parse_args()
    if args.max_tokens < 2 or args.warmup < 0 or args.requests < 1:
        parser.error("Require >=2 outputs, >=0 warmup and >=1 requests")
    if args.rounds_per_command < 1:
        parser.error("--rounds-per-command must be positive")
    if (args.target_graph or args.draft_graph) and args.warmup < 1:
        parser.error("Graph capture requires an unmeasured warmup request")
    if args.conditioned_graph and not args.draft_graph:
        parser.error("--conditioned-graph requires --draft-graph")
    if args.compile_target and not args.target_graph:
        parser.error("--compile-target requires --target-graph")
    if args.super_kernel and not args.compile_target:
        parser.error("--super-kernel requires --compile-target")
    if args.capacity < PREFIX_LENGTH + args.max_tokens + args.width - 2:
        parser.error("Capacity must permit a full final verification")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("Memory utilization must be in (0,1]")
    return args


def drive(backend, first, args):
    from tilert.pd_vllm.ascend_server import create_session

    timed = TimedBackend(backend)
    session = create_session(timed, args.width, (),
                             rounds_per_command=args.rounds_per_command)
    records = []
    try:
        for number in range(args.warmup + args.requests):
            tokens, clocks = [], []

            def deliver(token):
                tokens.append(int(token))
                clocks.append(perf_counter_ns())

            try:
                session.inject(PrefixRequest(f"mtp-bench-{number}",
                                             PREFIX_LENGTH))
                output = session.decode(
                    first, max_tokens=args.max_tokens,
                    sampling={"temperature": 0.0, "ignore_eos": True},
                    on_token=deliver,
                )
                records.append({
                    "warmup": number < args.warmup, "tokens": tokens,
                    "returned_tokens": output, "timestamps_ns": clocks,
                    **token_metrics(tokens, clocks),
                    "session": dict(session.last_stats),
                    "operations": timed.operations,
                })
            finally:
                if not backend.failed:
                    session.reset()
    finally:
        if not backend.failed:
            backend.close()
    for request in records:
        for operation in request["operations"]:
            operation["elapsed_ns"] = (
                operation["end_ns"] - operation["start_ns"]
            )
    return records


def validate_local(report, args):
    """Check saved local state, then return the rank-independent history."""
    histories = []
    if (report["final_positions"] != [0, 0] or not report["closed"]
            or len(report["rank_requests"]) != args.warmup + args.requests):
        raise ValueError("The rank did not reset and close every request")
    for request in report["rank_requests"]:
        tokens, position, previous_end = [request["first_token"]], 8192, 0
        rounds = []
        for item in request["rounds"]:
            candidates, predictions = item["candidates"], item["predictions"]
            matched = 0
            for proposed, predicted in zip(candidates[1:], predictions):
                if proposed != predicted:
                    break
                matched += 1
            accepted = predictions[:min(matched + 1,
                                        args.max_tokens - len(tokens))]
            if (item["width"] != args.width
                    or not len(candidates) == len(predictions)
                    == args.width
                    or candidates[0] != tokens[-1]
                    or item["accepted_tokens"] != accepted
                    or item["matched_drafts"] != matched
                    or item["natural_accepted"] != matched + 1
                    or item["before"] != position
                    or item["written"] != position + args.width
                    or item["retained"] != position + len(accepted)
                    or item["draft_retained"] != item["retained"]
                    or not previous_end <= item["start_ns"] <= item["end_ns"]
                    or item["elapsed_ns"]
                    != item["end_ns"] - item["start_ns"]):
                raise ValueError("Round acceptance/position/clock mismatch")
            tokens.extend(accepted)
            position, previous_end = item["retained"], item["end_ns"]
            rounds.append({key: value for key, value in item.items()
                           if not key.endswith("_ns")})
        if (request["injected"] != [8192, 8191]
                or request["finished"] != [position, position]
                or request["reset"] != [0, 0]
                or len(tokens) != args.max_tokens
                or position != 8192 + args.max_tokens - 1):
            raise ValueError("Request output and valid KV lengths differ")
        histories.append({"tokens": tokens, "rounds": rounds})
    expected_serial = 1 + sum(4 + len(item["rounds"]) for item in histories)
    if report["command_serial"] != expected_serial:
        raise ValueError("The Gloo command count differs from session rounds")
    if args.target_graph:
        graph = report["graph"]
        if (graph["width"] != args.width or graph["captures"] != 1
                or graph["replays"] != sum(len(h["rounds"]) for h in histories)
                or len(graph["entries"]) != 1
                or not graph["entries"][0]["has_aclgraph"]):
            raise ValueError("M4 graph capture/replay counts differ")
    if args.draft_graph:
        graph = report["draft_graph_state"]
        if (graph["captures"] != 1
                or graph["replays"] != (args.width - 2) * sum(
                    len(history["rounds"]) for history in histories)):
            raise ValueError("M1 draft graph capture/replay counts differ")
    if args.conditioned_graph:
        graph = report["conditioned_graph_state"]
        if (graph["captures"] != 1 or graph["replays"] != sum(
                len(history["rounds"]) for history in histories)):
            raise ValueError("M4 conditioned graph counts differ")
    if report["rank"] == 0:
        validate_delivery(report["requests"], histories, args)
    return histories


def validate_delivery(requests, histories, args):
    if len(requests) != len(histories):
        raise ValueError("Missing callback timing records")
    for request, history in zip(requests, histories):
        tokens, clocks = request["tokens"], request["timestamps_ns"]
        stats, rounds = request["session"], history["rounds"]
        operations = request["operations"]
        # The session drives rounds through advance_mtp_n, so an expectation
        # written against advance_mtp never matched and rank 0 raised on every
        # run -- including the one that recorded 14.247 ms.
        advances = [op for op in operations
                    if op["operation"] == "advance_mtp_n"]
        expected_ops = (["inject", "start_mtp"]
                        + ["advance_mtp_n"] * len(advances)
                        + ["finish_mtp", "reset"])
        if (tokens != request["returned_tokens"] or tokens != history["tokens"]
                or token_metrics(tokens, clocks)["tpot_ms"]
                != request["tpot_ms"]
                or [op["operation"] for op in operations] != expected_ops
                or stats["round_calls"] != stats["completed_rounds"]
                or stats["round_calls"] != len(rounds)
                or len(stats["accepted_spans"]) != len(rounds)
                or stats["verified_positions"]
                != args.width * len(rounds)
                or stats["accepted"] != args.max_tokens - 1
                or stats["output_tokens"] != args.max_tokens
                or stats["initial_position"] != PREFIX_LENGTH
                or stats["committed_position"] != 8192 + args.max_tokens - 1
                or stats["finish_reason"] != "length"
                or stats["finish_error"] is not None
                or len(advances) != len(rounds)
                or operations[0]["end_ns"] > clocks[0]
                or operations[1]["start_ns"] < clocks[0]
                or advances[-1]["end_ns"] > clocks[-1]
                or operations[-2]["start_ns"] < clocks[-1]):
            raise ValueError("Delivery clocks and session acceptance differ")
        for op, span, row in zip(advances, stats["accepted_spans"], rounds):
            count = len(row["accepted_tokens"])
            if (op["accepted_tokens"] != row["accepted_tokens"]
                    or not op["accepted"] == span["accepted"] == count
                    or span["output_tokens"] != count
                    or op["natural_accepted"] != row["natural_accepted"]
                    or span["natural_accepted"] != row["natural_accepted"]
                    or span["main_retained"] != row["retained"]):
                raise ValueError("Backend timing and delivered spans differ")
        previous = 0
        for op in operations:
            if not previous <= op["start_ns"] <= op["end_ns"]:
                raise ValueError("Backend operation clocks are not monotonic")
            previous = op["end_ns"]


def run(args, report):
    import torch
    import torch.distributed as distributed
    import torch_npu  # noqa: F401
    from vllm.distributed import get_world_group
    from vllm.compilation.counter import compilation_counter
    from tilert.models.glm_5_2_ascend.cann_model import CannModel
    from tilert.models.glm_5_2_ascend.cann_mtp import CannMtp
    from tilert.models.glm_5_2_ascend.tp_runtime import (
        CoordinatedBackend, RankExecutor, TorchControlChannel,
    )

    rank = report["rank"]
    if (int(os.environ["WORLD_SIZE"]) != 16
            or int(os.environ["LOCAL_RANK"]) != rank):
        raise ValueError("Launch one local sixteen-rank torchrun group")
    if any(os.environ.get(name) for name in (
        "PROFILER_SAMPLECONFIG", "PROFILING_MODE",
    )):
        raise ValueError("Run this measurement without profiling")
    torch.set_num_threads(1)
    checkpoint = checkpoint_fingerprint(args.checkpoint)
    main, mtp, parent, first, inputs = load_inputs(args, rank, checkpoint)
    if inputs["initial_position"] != PREFIX_LENGTH:
        raise ValueError("Use the captured full-P 8K prefix")
    target = CannModel(args.checkpoint, f"npu:{rank}", rank,
                       capacity=args.capacity,
                       gpu_memory_utilization=args.gpu_memory_utilization,
                       compile_target=args.compile_target,
                       super_kernel=args.super_kernel,
                       attention_kernel=args.attention_kernel,
                       fp32_head=args.fp32_head,
                       index_topk=args.index_topk,
                       expert_parallel=args.expert_parallel,
                       onesided_reduce=args.onesided_reduce,
                       graph_width=args.width,
                       hoist_metadata=args.hoist_metadata,
                       hoist_forward_context=args.hoist_forward_context)
    draft = None
    try:
        draft = CannMtp(target, shard_projections=args.draft_projection_tp)
        if not draft.model.do_not_compile:
            raise ValueError("MTP backbone compilation must be disabled")
        if "float32" not in target.tp_reduction:
            raise ValueError("Keep the production FP32 TP reduction")
        channel = TorchControlChannel(get_world_group().cpu_group)
        engine = MeasuredRank(target, draft, channel.group)
        prefix = FullRankPrefix(main, mtp, parent.to(target.device), 8192)
        with torch.inference_mode():
            if args.target_graph or args.draft_graph:
                counters_before = asdict(compilation_counter)
                engine.inject(prefix)
                engine.start_mtp(first, args.max_tokens,
                                 args.width, ())
                engine.advance_mtp()
                engine.finish_mtp(engine.position)
                engine.reset()
                warmup_key = ("compiled_graph_warmup" if args.compile_target
                              else "eager_graph_warmup")
                report[warmup_key] = engine.requests.pop()
                if args.compile_target:
                    backbone = target.model.model
                    config = target.config.compilation_config
                    cache = (Path(config.local_cache_dir)
                             if config.local_cache_dir else None)
                    artifacts = {}
                    if cache is not None:
                        index = cache / "vllm_compile_cache.py"
                        if index.is_file():
                            artifacts[str(index)] = digest(index)
                            for entry in ast.literal_eval(
                                index.read_text(),
                            ).values():
                                path = entry["graph_handle"][1]
                                if path and Path(path).is_file():
                                    artifacts[path] = digest(Path(path))
                    report["target_compilation"] = {
                        "mode": str(config.mode), "backend": config.backend,
                        "cudagraph_mode": str(config.cudagraph_mode),
                        "capture_sizes": config.cudagraph_capture_sizes,
                        "ascend_compilation_config": (
                            target.config.additional_config
                            ["ascend_compilation_config"]),
                        "backbone_compiled": backbone.compiled,
                        "aot_callable": callable(getattr(
                            backbone, "aot_compiled_fn", None,
                        )),
                        "aot_loaded": getattr(
                            backbone, "was_aot_compile_fn_loaded_from_disk",
                            False,
                        ),
                        "counters_before": counters_before,
                        "counters_after": asdict(compilation_counter),
                        "artifacts_sha256": artifacts,
                        "draft_do_not_compile": draft.model.do_not_compile,
                        "draft_mode": str(
                            draft.config.compilation_config.mode,
                        ),
                        "draft_cudagraph_mode": str(
                            draft.config.compilation_config.cudagraph_mode,
                        ),
                        "separate_context": (
                            draft.config.compilation_config
                            .static_forward_context is not
                            config.static_forward_context),
                    }
                if args.target_graph:
                    target.enable_graph(width=args.width)
                if args.draft_graph:
                    draft.enable_graph(
                        conditioned_width=(args.width
                                           if args.conditioned_graph else 0))
            sources = source_hashes(target, draft)
            for name in ("mtp_session.py", "tp_runtime.py"):
                path = ROOT / "tilert/models/glm_5_2_ascend" / name
                sources[str(path)] = digest(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("bench_cann_decode.py"),
                ROOT / "tilert/pd_vllm/decode_timing.py",
                ROOT / "tilert/pd_vllm/ascend_server.py",
                ROOT / "tilert/models/glm_5_2_ascend/session.py",
            ):
                sources[str(path)] = digest(path)
            report.update(
                inputs=inputs, checkpoint_sha256=checkpoint,
                source_sha256=sources, model_tp_reduction=target.tp_reduction,
                packages={name: version(name) for name in
                          ("torch", "torch-npu", "vllm", "vllm-ascend")},
                device=str(target.device), runtime_env={
                    name: os.environ.get(name) for name in (
                        "HCCL_OP_EXPANSION_MODE", "HCCL_BUFFSIZE",
                        "HCCL_DETERMINISTIC", "TASK_QUEUE_ENABLE",
                        "ASCEND_LAUNCH_BLOCKING", "ASCEND_CUSTOM_OPP_PATH",
                        "ASCEND_RT_VISIBLE_DEVICES",
                        # The launch recipe is not in the command. Leaving the
                        # autograd cache on fails after the weights have
                        # loaded, a shared cache root replays another arm's
                        # compiled graph, and a bound HCCL port range refuses
                        # the next arm -- three launches each.
                        "TORCHINDUCTOR_AUTOGRAD_CACHE", "VLLM_CACHE_ROOT",
                        "HCCL_NPU_SOCKET_PORT_RANGE")},
            )
            if args.cpu_affinity:
                bind_cpu_threads(rank, target.device, report)
            executor = RankExecutor(engine, rank,
                                    lambda request, _rank: prefix)
            if rank == 0:
                backend = CoordinatedBackend(executor, channel)
                report["requests"] = drive(backend, first, args)
            else:
                executor.serve(channel)
        graph = target.graph
        report.update(
            status="completed", rank_requests=engine.requests,
            command_serial=executor.serial, closed=executor.closed,
            final_positions=[target.position, draft.position],
            draft_graph_state=None if draft.graph is None else {
                "captures": draft.graph.capture_count,
                "replays": draft.graph.replay_count,
                "scope": "M1 shared-index chain, including own head",
            },
            conditioned_graph_state=(
                None if draft.conditioned_graph is None else {
                    "captures": draft.conditioned_graph.capture_count,
                    "replays": draft.conditioned_graph.replay_count,
                    "scope": "M4 target-conditioned pass, including own head",
                }),
            graph=None if graph is None else {
                "width": graph.width, "captures": graph.capture_count,
                "replays": graph.replay_count,
                "entries": [{"descriptor": str(key),
                             "has_aclgraph": entry.aclgraph is not None,
                             "input_addresses": entry.input_addresses}
                            for key, entry in graph.wrapper
                            .concrete_aclgraph_entries.items()]},
        )
        path = args.output / f"rank-{rank}.json"
        atomic_json(path, report)
        errors, histories = [], None
        try:
            histories = validate_local(report, args)
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
        receipts = [None] * 16
        distributed.all_gather_object(receipts, {
            "rank": rank, "report_sha256": digest(path), "errors": errors,
            "histories": histories, "source": report["source_sha256"],
            "checkpoint": checkpoint, "packages": report["packages"],
            "prompt": inputs["prompt_sha256"],
            "capture": inputs["capture_manifest_sha256"],
            "parent": inputs["parent_hidden_sha256"],
        }, group=channel.group)
        same = all(all(item[key] == receipts[0][key] for key in (
            "histories", "source", "checkpoint", "packages", "prompt",
            "capture", "parent")) for item in receipts)
        passed = (
            same and [item["rank"] for item in receipts] == list(range(16))
            and not any(item["errors"] for item in receipts)
        )
        if rank == 0:
            measured = [item for item in report["requests"]
                        if not item["warmup"]]
            values = [item["tpot_ms"] for item in measured]
            summary = {
                "status": "measured" if passed else "fail",
                "all_rank_state_agreement": same,
                "precision_verified": False, "formal_1024_token_run": False,
                "tpot_ms": {"median": statistics.median(values),
                            "mean": statistics.mean(values),
                            "p95": statistics.quantiles(
                                values, n=100, method="inclusive",
                            )[94] if len(values) > 1 else values[0],
                            "min": min(values), "max": max(values)},
                "warmup": args.warmup, "requests": args.requests,
                "output_tokens_per_request": args.max_tokens,
                "target_graph": args.target_graph,
                "draft_graph": args.draft_graph,
                "conditioned_graph": args.conditioned_graph,
                "draft_projection_tp": args.draft_projection_tp,
                "hoist_metadata": args.hoist_metadata,
                "hoist_forward_context": args.hoist_forward_context,
                "compile_target": args.compile_target,
                "super_kernel": args.super_kernel,
                "attention_kernel": args.attention_kernel,
                "fp32_head": args.fp32_head,
                "ranks": [{key: item[key] for key in
                           ("rank", "report_sha256", "errors")}
                          for item in receipts],
            }
            atomic_json(args.output / "summary.json", summary)
        return passed
    finally:
        if draft is not None:
            draft.close()
        target.shutdown()


def main():
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "error", "rank": int(os.environ.get("RANK", "0")),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv], "hostname": platform.node(),
        "source_commit": args.source_commit, "world_size": 16,
        "width": args.width, "max_tokens": args.max_tokens,
        "capacity": args.capacity, "warmup": args.warmup,
        "measured_requests": args.requests, "target_graph": args.target_graph,
        "draft_graph": args.draft_graph,
        "conditioned_graph": args.conditioned_graph,
        "draft_projection_tp": args.draft_projection_tp,
        "hoist_metadata": args.hoist_metadata,
        "hoist_forward_context": args.hoist_forward_context,
        "compile_target": args.compile_target,
        "super_kernel": args.super_kernel,
        "attention_kernel": args.attention_kernel,
        "fp32_head": args.fp32_head,
        "scope": "internal TP16 session delivery; HTTP/HIXL excluded",
        "clock": "perf_counter_ns; rank-local host clocks",
        "tpot_formula": "(last_callback-first_callback)/(output_tokens-1)",
        "prefix_readiness": "Local synchronize then Gloo barrier before g0",
        "round_scope": "Gloo, draft chain, verify, MTP recompute, commit",
        "phase_device_timing": False,
        "draft_execution": "chain_graph" if args.draft_graph else "eager",
        "conditioned_execution": (
            "conditioned_graph" if args.conditioned_graph else "eager"),
        "ignore_eos": True, "precision_verified": False,
        "formal_1024_token_run": False, "profiled": False,
    }
    try:
        passed = run(args, report)
    except Exception:
        passed = False
        report["status"], report["error"] = "error", traceback.format_exc()
        atomic_json(args.output / f"rank-{report['rank']}.json", report)
    print(json.dumps({"rank": report["rank"], "passed": passed,
                      "output": str(args.output)}))
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
