"""Capture two CANN M1 forwards after all ranks pass numerical validation."""

import inspect
import json
from pathlib import Path

from profile_native_session import (
    digest, make_profiler, npu_range, profiler_identity, trace_inventory,
)
from tilert.models.glm_5_2_ascend.profiling import (
    StageMarkers, capture_phase, reject_msprof_environment, run_phase,
)
from tilert.models.glm_5_2_ascend.session import AscendPDEngine, PrefixRequest
from tilert.models.glm_5_2_ascend.tp_runtime import TorchControlChannel
from tilert.pd_vllm.ascend_server import create_session


class _MarkedModel:
    def __init__(self, model, markers, first_token):
        self.model = model
        self.markers = markers
        self.first_token_id = first_token

    def __getattr__(self, name):
        return getattr(self.model, name)

    def inject(self, prefix):
        with self.markers.span("inject"):
            return self.model.inject(prefix)

    def step(self, token):
        with self.markers.span("step", position=self.position, token=token):
            return self.model.step(token)

    def reset(self):
        with self.markers.span("reset"):
            return self.model.reset()


def capture_validated_model(engine, prefixes, first_token, rank, directory):
    """Profile the caller's accepted model without reloading or prompt replay.

    All sixteen ranks must call this after their numerical checks pass.
    The caller binds those reports and the frozen prefix to this receipt.
    """
    import torch
    import torch.distributed as distributed
    import torch_npu
    from vllm.distributed import get_world_group

    reject_msprof_environment()
    if (len(prefixes) != 78 or prefixes[0]["latent"].shape[0] != 8192
            or engine.capacity < 8194):
        raise ValueError("Capture requires full-P 8K KV and two decode rows")
    if type(first_token) is not int or not 0 <= first_token < 154880:
        raise ValueError("Capture requires the prefill vocabulary token")
    if distributed.get_rank() != rank or distributed.get_world_size() != 16:
        raise ValueError("Capture requires all sixteen TP ranks")
    identity = profiler_identity(torch_npu)
    directory = Path(directory)
    raw = directory / f"raw-rank-{rank}"
    raw.mkdir(parents=True, exist_ok=False)
    markers = StageMarkers(rank)
    model = _MarkedModel(engine, markers, first_token)
    channel = TorchControlChannel(get_world_group().cpu_group)
    request = PrefixRequest("capture", 8192)
    phase_args = (model, rank, channel, lambda request, rank: prefixes,
                  request, 1, 3, markers)

    def barrier():
        distributed.barrier(group=channel.group)

    with torch.npu.device(engine.device), torch.inference_mode():
        warmup = run_phase(*phase_args, requests=3, ignore_eos=True)
        torch.npu.synchronize(engine.device)
        markers.phase = "capture"
        markers.range_factory = npu_range
        captured = capture_phase(
            make_profiler(raw, rank), markers, barrier,
            lambda: run_phase(*phase_args, ignore_eos=True),
        )
    consistent = [True]
    if rank == 0:
        records = warmup["requests"] + captured["requests"]
        expected = records[0]["tokens"]
        consistent[0] = (len(expected) == 3 and all(
            item["tokens"] == expected and item["step_calls"] == 2
            and item["committed_position"] == 8194 for item in records
        ))
    distributed.broadcast_object_list(consistent, src=0, group=channel.group)
    objects = (run_phase, make_profiler, TorchControlChannel,
               AscendPDEngine, create_session)
    sources = {str(Path(inspect.getfile(value)).resolve()): digest(
        Path(inspect.getfile(value))) for value in objects}
    sources[str(Path(__file__).resolve())] = digest(Path(__file__))
    report = {
        "status": "raw_capture" if consistent[0] else "token_mismatch",
        "rank": rank, "world_size": 16, "device": str(engine.device),
        "scope": "Full-P CANN M1 session; HTTP and HIXL excluded",
        "profiled": True, "tpot_measured": False, "mtp_width": 1,
        "initial_position": 8192, "first_token_id": first_token,
        "max_tokens": 3, "ignore_eos": True,
        "profiler_api": identity, "warmup": warmup, "capture": captured,
        "markers": markers.events, "trace_directory": str(raw),
        "source_sha256": {**engine.source_metadata(), **sources},
        "trace_files": trace_inventory(raw),
    }
    (directory / f"rank-{rank}.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
    )
    if not consistent[0]:
        raise RuntimeError("Profiled request differs from warmup replay")
    return report
