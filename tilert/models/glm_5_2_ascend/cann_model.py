"""M1 GLM-5.2 execution through the pinned vLLM-Ascend model and CANN."""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re

from tilert.pd_vllm.ascend_layout import (
    FULL_INDEX_LAYERS, INT8_INDEX_LAYERS, PAGE_SIZE, cache_layout,
)


@dataclass
class CannStep:
    logits: object
    last_hidden: object
    token_ids: object


class CannModel:
    """One request, TP16, full-P injection and one-token decode forwards.

    ``rank`` is the logical TP rank; ``device`` selects the process's NPU.
    The caller supplies all 16 visible devices without a visibility rotation.
    Worker construction and loading execute device code. No scheduler,
    speculative model, profiling pass, or graph capture is created here.
    """

    def __init__(
        self, checkpoint, device, rank, init_method="env://", capacity=9344,
        gpu_memory_utilization=0.85,
    ):
        if type(rank) is not int or not 0 <= rank < 16:
            raise ValueError("GLM-5.2 requires a logical TP rank in [0, 16)")
        if capacity < 128:
            raise ValueError("Capacity must be at least one 128-token page")
        if int(os.environ.get("WORLD_SIZE", "16")) != 16:
            raise ValueError("Launch exactly 16 ranks")
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"

        import torch
        import torch_npu  # noqa: F401
        from vllm.config import set_current_vllm_config
        from vllm.engine.arg_utils import EngineArgs
        from vllm.platforms import current_platform
        from vllm_ascend.worker.worker import NPUWorker

        self.device = torch.device(device)
        if self.device.type != "npu" or self.device.index not in range(16):
            raise ValueError("device must name one of the 16 visible NPUs")
        self.rank = rank
        self.capacity = capacity
        self.position = 0
        self.checkpoint = str(Path(checkpoint).resolve())
        torch.npu.set_device(self.device)
        current_platform.pre_register_and_update()
        self.config = EngineArgs(
            model=self.checkpoint, trust_remote_code=True,
            dtype="bfloat16", quantization="ascend",
            tensor_parallel_size=16, pipeline_parallel_size=1,
            data_parallel_size=1,
            distributed_executor_backend="external_launcher",
            attention_config={"indexer_kv_dtype": "int8"},
            kv_cache_dtype="auto", block_size=PAGE_SIZE,
            max_num_seqs=1, max_num_batched_tokens=16,
            max_model_len=capacity,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=False, enable_chunked_prefill=True,
            enforce_eager=True, async_scheduling=False,
            speculative_config=None, kv_transfer_config=None,
            additional_config={
                "enable_flashcomm1": False, "enable_dsa_cp": False,
                "enable_shared_expert_dp": False, "enable_fused_mc2": 0,
            },
        ).create_engine_config()
        with set_current_vllm_config(self.config):
            self.worker = NPUWorker(
                self.config, local_rank=self.device.index, rank=rank,
                distributed_init_method=init_method,
                is_driver_worker=rank == 0,
            )
            self.worker.init_device()
            if self.worker.device != self.device:
                raise ValueError("Worker remapped the requested physical NPU")
            self.worker.load_model()
            self.runner = self.worker.model_runner
            self.model = self.runner.model
            if self.runner.drafter is not None:
                raise ValueError("The M1 adapter must not load a draft model")
            self._bind_cache()
        self._allocate_inputs()

    def _bind_cache(self):
        import torch
        from vllm_ascend.attention.indexer import (
            AscendSFAIndexerMetadataBuilder,
        )
        from vllm_ascend.attention.sfa_v1 import AscendSFAMetadataBuilder
        from vllm_ascend.core.kv_cache_interface import (
            AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec,
        )
        from vllm_ascend.patch.worker.patch_bind_kv_cache import (
            bind_kv_cache_to_layers,
        )

        self.layout = cache_layout(self.capacity)
        self.storage = torch.empty(
            self.layout["nbytes"], dtype=torch.uint8, device=self.device,
        )
        self.views = {}
        for name, region in self.layout["regions"].items():
            start = region["offset"]
            self.views[name] = self.storage[
                start:start + region["nbytes"]
            ].view(getattr(torch, region["dtype"])).view(region["shape"])
        self.cache = []
        for layer in range(78):
            entry = {
                "latent": self.views[f"mla_latent.{layer}"].view(-1, 512),
                "rope": self.views[f"mla_rope.{layer}"].view(-1, 64),
            }
            if layer in FULL_INDEX_LAYERS:
                entry["index_keys"] = self.views[
                    f"index.{layer}"
                ].view(-1, 128)
            if layer in INT8_INDEX_LAYERS:
                entry["index_scales"] = self.views[
                    f"index_scale.{layer}"
                ].view(-1)
            self.cache.append(entry)

        groups, caches = {}, {}
        main_layers, index_layers = set(), set()
        for name, spec in self.runner.get_kv_cache_spec().items():
            match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
            if match is None or int(match[1]) >= 78:
                raise ValueError(f"Unexpected main cache layer: {name}")
            layer = int(match[1])
            if isinstance(spec, AscendSFAIndexerCacheSpec):
                index_layers.add(layer)
                values = (self.views[f"index.{layer}"],)
                if layer in INT8_INDEX_LAYERS:
                    values += (
                        self.views[f"index_scale.{layer}"].unsqueeze(-1),
                    )
                builder_type = AscendSFAIndexerMetadataBuilder
            elif isinstance(spec, AscendMLAAttentionSpec):
                main_layers.add(layer)
                values = (
                    self.views[f"mla_latent.{layer}"],
                    self.views[f"mla_rope.{layer}"],
                )
                builder_type = AscendSFAMetadataBuilder
            else:
                raise ValueError(f"Unexpected KV specification: {spec}")
            caches[name] = values
            groups.setdefault((builder_type, spec), []).append(name)
        if main_layers != set(range(78)) or index_layers != {
            layer for layer in FULL_INDEX_LAYERS if layer < 78
        }:
            raise ValueError("Expected 78 main and 21 indexer cache layers")
        bind_kv_cache_to_layers(
            caches, self.config.compilation_config.static_forward_context,
        )
        self.builders = [
            (names, kind(spec, names, self.config, self.device))
            for (kind, spec), names in groups.items()
        ]
        self.cache_names = sorted(caches)

    def _allocate_inputs(self):
        import torch

        self.input_ids = torch.empty(1, dtype=torch.long, device=self.device)
        self.positions = torch.empty_like(self.input_ids)
        self.slots = torch.empty(1, dtype=torch.int32, device=self.device)
        self.seq_lens = torch.empty_like(self.slots)
        self.query_cpu = torch.tensor([0, 1], dtype=torch.int32)
        self.query = self.query_cpu.to(self.device)
        self.block_table = torch.arange(
            self.layout["capacity_tokens"] // PAGE_SIZE,
            dtype=torch.int32, device=self.device,
        ).unsqueeze(0)

    def inject(self, prefixes):
        """Copy all P main rows; the next consumed token has position P."""
        import torch

        if len(prefixes) != 78:
            raise ValueError("Full-P injection requires all 78 main layers")
        length = prefixes[0]["latent"].shape[0]
        if not 0 <= length < self.capacity:
            raise ValueError("Prefix leaves no room for a decode token")
        for layer, (source, target) in enumerate(zip(prefixes, self.cache)):
            source = {key: value for key, value in source.items()
                      if value is not None}
            if source.keys() != target.keys():
                raise ValueError(f"Layer {layer}: cache fields differ")
            for name, destination in target.items():
                value = source[name]
                if (value.shape != destination[:length].shape
                        or value.dtype != destination.dtype):
                    raise ValueError(f"Layer {layer}: invalid {name} cache")
        with torch.npu.device(self.device), torch.inference_mode():
            for source, target in zip(prefixes, self.cache):
                for name, destination in target.items():
                    destination[:length].copy_(source[name])
        self.position = length

    def _metadata(self):
        import torch
        from vllm_ascend.attention.attention_v1 import AscendAttentionState
        from vllm_ascend.attention.utils import AscendCommonAttentionMetadata

        length_cpu = torch.tensor([self.position + 1], dtype=torch.int32)
        computed_cpu = torch.tensor([self.position], dtype=torch.int32)
        self.seq_lens.copy_(length_cpu)
        self.slots.fill_(self.position)
        self.positions.fill_(self.position)
        common = AscendCommonAttentionMetadata(
            query_start_loc=self.query, query_start_loc_cpu=self.query_cpu,
            seq_lens=self.seq_lens, seq_lens_cpu=length_cpu,
            seq_lens_cpu_upper_bound=length_cpu,
            num_computed_tokens_cpu=computed_cpu,
            _seq_lens_cpu=length_cpu,
            _num_computed_tokens_cpu=computed_cpu,
            num_reqs=1, num_actual_tokens=1, max_query_len=1,
            max_seq_len=self.position + 1,
            block_table_tensor=self.block_table, slot_mapping=self.slots,
            positions=self.positions, positions_cpu=computed_cpu.long(),
            is_prefilling=torch.tensor([False]),
            decode_token_per_req=1, actual_seq_lengths_q=[1],
            attn_state=AscendAttentionState.DecodeOnly, num_input_tokens=1,
        )
        result = {}
        for names, builder in self.builders:
            metadata = builder.build(0, common)
            result.update((name, metadata) for name in names)
        return result

    def verify(self, tokens, *, full_logits=True):
        """Consume one pending token and return its next-token logits."""
        import torch
        from vllm.config import set_current_vllm_config
        from vllm_ascend.ascend_forward_context import (
            set_ascend_forward_context,
        )

        if (len(tokens) != 1 or type(tokens[0]) is not int
                or not 0 <= tokens[0] < 154880):
            raise ValueError("This adapter accepts one vocabulary token")
        if self.position >= self.capacity:
            raise ValueError("Decode exceeds cache capacity")
        with torch.npu.device(self.device), torch.inference_mode(), (
            set_current_vllm_config(self.config)
        ):
            self.input_ids.fill_(tokens[0])
            metadata = self._metadata()
            with set_ascend_forward_context(
                metadata, self.config, num_tokens=1, num_actual_tokens=1,
                model_instance=self.model,
            ):
                hidden = self.model(self.input_ids, self.positions)
                logits = self.model.compute_logits(hidden)
                token_ids = logits.argmax(dim=-1)
            self.position += 1
            return CannStep(
                logits if full_logits else None, hidden, token_ids,
            )

    def step(self, token):
        return self.verify((token,), full_logits=False)

    def reset(self):
        """Exclude prior rows from attention without replaying the prompt."""
        self.position = 0

    def source_metadata(self):
        import inspect
        from vllm_ascend.ascend_forward_context import (
            set_ascend_forward_context,
        )

        objects = [type(self), type(self.worker), type(self.runner),
                   type(self.model), type(self.model.model),
                   set_ascend_forward_context]
        objects.extend(type(builder) for _, builder in self.builders)
        paths = {Path(inspect.getfile(inspect.unwrap(value))).resolve()
                 for value in objects}
        return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(paths)}
