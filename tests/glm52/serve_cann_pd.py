"""Serve M1 CANN model decode from a complete vLLM/HIXL main prefix."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def run(args):
    import torch
    import torch.distributed as distributed
    import torch_npu  # noqa: F401
    import uvicorn

    from tilert.models.glm_5_2_ascend.cann_model import CannModel
    from tilert.models.glm_5_2_ascend.session import AscendPDEngine
    from tilert.models.glm_5_2_ascend.tp_runtime import (
        CoordinatedBackend, RankExecutor, TorchControlChannel,
    )
    from tilert.pd_vllm import decode_server
    from tilert.pd_vllm.ascend_cache import (
        AscendReceiveBuffer, checkpoint_fingerprint, receive_device_id,
    )
    from tilert.pd_vllm.ascend_receiver import AscendReceiveServer
    from tilert.pd_vllm.ascend_server import AscendDeviceContext
    from tilert.pd_vllm.decode_timing import atomic_json
    from tilert.pd_vllm.hixl_transport import HixlTransport
    from tilert.pd_vllm.prefix_capture import digest

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if rank != local_rank:
        raise ValueError("M1 PD serving requires one local TP16 group")
    device = torch.device(f"npu:{receive_device_id(local_rank)}")
    torch.set_num_threads(1)
    model = CannModel(
        args.checkpoint, device, rank, capacity=args.capacity,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    transport = HixlTransport(
        args.hixl_library, port=args.hixl_port_base + rank,
    )
    receive = AscendReceiveBuffer(
        rank, device, args.capacity, transport, args.hixl_host,
    )
    endpoints = [None] * 16
    distributed.all_gather_object(endpoints, receive.endpoint())
    channel = TorchControlChannel()
    executor = RankExecutor(model, rank, lambda request, _rank:
                            receive.main_prefix(request))
    fingerprint = checkpoint_fingerprint(args.checkpoint)
    sources = [Path(__file__),
               ROOT / "tilert/models/glm_5_2_ascend/session.py",
               ROOT / "tilert/models/glm_5_2_ascend/tp_runtime.py",
               *(ROOT / "tilert/pd_vllm").glob("*.py")]
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "engine": "CannModel", "rank": rank, "device": str(device),
        "scope": "M1 HTTP/HIXL continuation from full-P main KV",
        "precision_verified": False, "performance_measured": False,
        "checkpoint_sha256": fingerprint,
        "source_sha256": {**model.source_metadata(),
                          **{str(path): digest(path) for path in sources}},
        "hixl_library_sha256": digest(args.hixl_library),
        "world_size": distributed.get_world_size(),
        "capacity": model.capacity,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "command": [sys.executable, *sys.argv], "status": "initialized",
    }
    report = args.output / f"rank-{rank}.json"
    atomic_json(report, metadata)
    if rank == 0:
        config = json.loads((args.checkpoint / "config.json").read_text())
        stop_ids = config["eos_token_id"]
        if type(stop_ids) is int:
            stop_ids = [stop_ids]
        backend = CoordinatedBackend(executor, channel)
        session = AscendPDEngine(
            backend, stop_ids=stop_ids,
            select_token=lambda result: int(result.token_ids[0].item()),
        )
        engine = AscendDeviceContext(session, device)
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
    model.worker.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hixl-host", required=True)
    parser.add_argument("--hixl-library", type=Path, required=True)
    parser.add_argument("--hixl-port-base", type=int, default=26100)
    parser.add_argument("--capacity", type=int, default=9344)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--ctrl-port", type=int, default=5556)
    parser.add_argument("--http-port", type=int, default=18080)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
