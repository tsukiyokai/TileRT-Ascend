"""Serve CANN M1 or MTP decode from full-P vLLM/HIXL prefixes."""

import argparse
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class DeferredModel:
    """Keep the live prefix on device until the model-loading gate exists."""

    def __init__(self, args, device, rank):
        self.args = args
        self.device = device
        self.rank = rank
        self.capacity = args.capacity
        self.model = None
        self.on_load = None

    @property
    def position(self):
        return 0 if self.model is None else self.model.position

    def inject(self, prefix):
        if self.model is None:
            from tilert.models.glm_5_2_ascend.cann_model import CannModel

            while not self.args.load_model_after.exists():
                time.sleep(0.1)
            self.model = CannModel(
                self.args.checkpoint, self.device, self.rank,
                capacity=self.capacity,
                gpu_memory_utilization=self.args.gpu_memory_utilization,
            )
            if self.on_load is not None:
                self.on_load(self.model)
        return self.model.inject(prefix)

    def step(self, token):
        return self.model.step(token)

    def reset(self):
        if self.model is not None:
            self.model.reset()

    def source_metadata(self):
        return {} if self.model is None else self.model.source_metadata()

    def shutdown(self):
        if self.model is not None:
            self.model.shutdown()
        else:
            import torch.distributed as distributed

            distributed.destroy_process_group()


def run(args):
    import torch
    import torch.distributed as distributed
    import torch_npu  # noqa: F401
    import uvicorn

    from tilert.models.glm_5_2_ascend.cann_model import CannModel
    from tilert.models.glm_5_2_ascend.tp_runtime import (
        CoordinatedBackend, RankExecutor, TorchControlChannel,
    )
    from tilert.pd_vllm import decode_server
    from tilert.pd_vllm.ascend_cache import (
        AscendReceiveBuffer, CannRankEngine, checkpoint_fingerprint,
        receive_device_id,
    )
    from tilert.pd_vllm.ascend_receiver import AscendReceiveServer
    from tilert.pd_vllm.ascend_server import (
        AscendDeviceContext, AscendServingEngine, create_session,
    )
    from tilert.pd_vllm.decode_timing import (
        TimedBackend, TimedEngine, atomic_json,
    )
    from tilert.pd_vllm.hixl_transport import HixlTransport
    from tilert.pd_vllm.prefix_capture import digest

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if rank != local_rank:
        raise ValueError("CANN PD serving requires one local TP16 group")
    device = torch.device(f"npu:{receive_device_id(local_rank)}")
    torch.set_num_threads(1)
    if args.load_model_after is not None:
        torch.npu.set_device(device)
        distributed.init_process_group("hccl")
        if distributed.get_world_size() != 16:
            raise ValueError("Deferred CANN serving requires TP16")
        model = DeferredModel(args, device, rank)
    else:
        model = CannModel(
            args.checkpoint, device, rank, capacity=args.capacity,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    rank_engine = model
    draft = None
    if args.mtp_width != 1:
        from tilert.models.glm_5_2_ascend.cann_mtp import CannMtp

        draft = CannMtp(model)
        rank_engine = CannRankEngine(model, draft)
    transport = HixlTransport(
        args.hixl_library, port=args.hixl_port_base + rank,
    )
    receive = AscendReceiveBuffer(
        rank, device, args.capacity, transport, args.hixl_host,
        parent_hidden=draft is not None,
    )
    endpoints = [None] * 16
    distributed.all_gather_object(endpoints, receive.endpoint())
    channel = TorchControlChannel()
    prefix_loader = (receive.main_prefix if draft is None
                     else receive.mtp_prefix)
    executor = RankExecutor(rank_engine, rank, lambda request, _rank:
                            prefix_loader(request))
    fingerprint = checkpoint_fingerprint(args.checkpoint)
    sources = [Path(__file__),
               *(ROOT / "tilert/models/glm_5_2_ascend").glob("*.py"),
               *(ROOT / "tilert/pd_vllm").glob("*.py")]
    if draft is not None:
        sources.append(Path(inspect.getfile(type(draft.model))))
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "engine": "CannModel", "rank": rank, "device": str(device),
        "scope": "HTTP/HIXL continuation from full-P main KV",
        "mtp_width": args.mtp_width,
        "parent_hidden": draft is not None,
        "precision_verified": False, "performance_measured": False,
        "checkpoint_sha256": fingerprint,
        "source_sha256": {**model.source_metadata(),
                          **{str(path): digest(path) for path in sources}},
        "hixl_library_sha256": digest(args.hixl_library),
        "world_size": distributed.get_world_size(),
        "capacity": model.capacity,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "load_model_after": (str(args.load_model_after)
                             if args.load_model_after is not None else None),
        "command": [sys.executable, *sys.argv], "status": "initialized",
    }
    report = args.output / f"rank-{rank}.json"
    if isinstance(model, DeferredModel):
        def record_model_sources(loaded):
            metadata["source_sha256"].update(loaded.source_metadata())
            atomic_json(report, metadata)

        model.on_load = record_model_sources
    atomic_json(report, metadata)
    if rank == 0:
        config = json.loads((args.checkpoint / "config.json").read_text())
        stop_ids = config["eos_token_id"]
        if type(stop_ids) is int:
            stop_ids = [stop_ids]
        backend = CoordinatedBackend(executor, channel)
        if args.timing_dir is not None:
            backend = TimedBackend(backend)
        session = create_session(backend, args.mtp_width, stop_ids)
        if draft is None:
            engine = AscendDeviceContext(session, device)
        else:
            engine = AscendServingEngine(session, rank_engine, device)
        if args.timing_dir is not None:
            engine = TimedEngine(engine, backend, args.timing_dir,
                                 metadata=metadata)
        server = AscendReceiveServer(
            receive.layout, endpoints, fingerprint, host=args.bind_host,
            ctrl_port=args.ctrl_port,
        )
        try:
            uvicorn.run(decode_server.build_app(server, engine),
                        host=args.bind_host, port=args.http_port)
        finally:
            server.close()
            with torch.npu.device(device):
                backend.close()
    else:
        executor.serve(channel)
    distributed.barrier()
    transport.close()
    metadata.update(status="stopped", command_serial=executor.serial,
                    final_position=model.position)
    atomic_json(report, metadata)
    model.shutdown()


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hixl-host", required=True)
    parser.add_argument("--hixl-library", type=Path, required=True)
    parser.add_argument("--hixl-port-base", type=int, default=26100)
    parser.add_argument("--capacity", type=int, default=9344)
    parser.add_argument("--mtp-width", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--ctrl-port", type=int, default=5556)
    parser.add_argument("--http-port", type=int, default=18080)
    parser.add_argument("--timing-dir", type=Path)
    parser.add_argument("--load-model-after", type=Path,
                        help="Load M1 weights after this gate file exists")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.load_model_after is not None and args.mtp_width != 1:
        parser.error("--load-model-after requires --mtp-width 1")
    return args


def main():
    run(parse_arguments())


if __name__ == "__main__":
    main()
