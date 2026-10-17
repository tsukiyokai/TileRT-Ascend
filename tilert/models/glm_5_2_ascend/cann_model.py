"""GLM-5.2 execution through the pinned vLLM-Ascend model and CANN."""

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


def create_config(checkpoint, capacity=9344, gpu_memory_utilization=0.85,
                  *, compile_target=False):
    """Construct TP16 config, optionally compiling the M4 target backbone."""
    from vllm.config import CompilationMode, CUDAGraphMode
    from vllm.engine.arg_utils import EngineArgs
    from vllm.platforms import current_platform

    current_platform.pre_register_and_update()
    return EngineArgs(
        model=str(Path(checkpoint).resolve()), trust_remote_code=True,
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
        enforce_eager=not compile_target, async_scheduling=False,
        compilation_config={
            "mode": CompilationMode.VLLM_COMPILE,
            "cudagraph_mode": CUDAGraphMode.FULL_DECODE_ONLY,
            "cudagraph_capture_sizes": [4],
            "max_cudagraph_capture_size": 4,
            "custom_ops": ["all"],
        } if compile_target else {},
        speculative_config=None, kv_transfer_config=None,
        additional_config={
            "enable_flashcomm1": False, "enable_dsa_cp": False,
            "enable_shared_expert_dp": False, "enable_fused_mc2": 0,
            "ascend_compilation_config": {
                "enable_static_kernel": False, "enable_super_kernel": False,
            },
        },
    ).create_engine_config()


class CannModel:
    """One request, TP16, full-P injection and M=1/2/4 target forwards.

    ``rank`` is the logical TP rank; ``device`` selects the process's NPU.
    The caller supplies all 16 visible devices without a visibility rotation.
    Worker construction and loading execute device code. No scheduler,
    speculative model, profiling pass, or graph capture is created here.
    """

    def __init__(
        self, checkpoint, device, rank, init_method="env://", capacity=9344,
        gpu_memory_utilization=0.85, *, compile_target=False,
        attention_kernel="sfa", fp32_head=False,
    ):
        if type(rank) is not int or not 0 <= rank < 16:
            raise ValueError("GLM-5.2 requires a logical TP rank in [0, 16)")
        if capacity < 128:
            raise ValueError("Capacity must be at least one 128-token page")
        if int(os.environ.get("WORLD_SIZE", "16")) != 16:
            raise ValueError("Launch exactly 16 ranks")
        if attention_kernel not in ("sfa", "gather_fia"):
            raise ValueError("attention_kernel must be sfa or gather_fia")
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"

        import torch
        import torch_npu  # noqa: F401
        from vllm.config import set_current_vllm_config
        from vllm_ascend.worker.worker import NPUWorker

        self.device = torch.device(device)
        if self.device.type != "npu" or self.device.index not in range(16):
            raise ValueError("device must name one of the 16 visible NPUs")
        self.rank = rank
        self.capacity = capacity
        self.position = 0
        self.attention_kernel = attention_kernel
        self.fp32_head = fp32_head
        self.checkpoint = str(Path(checkpoint).resolve())
        torch.npu.set_device(self.device)
        self.config = create_config(
            self.checkpoint, capacity, gpu_memory_utilization,
            compile_target=compile_target,
        )
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
            self.model = self.runner.get_model()
            if self.fp32_head:
                from .cann_head import install_fp32_head

                install_fp32_head(self.model.lm_head,
                                  self.model.logits_processor)
            if self.runner.drafter is not None:
                raise ValueError("Target must not load a draft model")
            self._bind_cache()
        self._allocate_inputs()
        self._install_tp_sum()

    def _install_tp_sum(self):
        from vllm.distributed import get_tp_group

        self._tp_sum_group = get_tp_group()
        self._tp_sum_original = self._tp_sum_group._all_reduce_out_place
        self._tp_sum_had_override = (
            "_all_reduce_out_place" in vars(self._tp_sum_group)
        )
        self._tp_sum_group._all_reduce_out_place = self._tp_reduce
        self.tp_reduction = "float32"

    def _tp_reduce(self, value):
        import torch

        if value.dtype == torch.bfloat16:
            return self._tp_sum_original(value.float()).bfloat16()
        return self._tp_sum_original(value)

    def _restore_tp_sum(self):
        if self._tp_sum_group is None:
            return
        if self._tp_sum_had_override:
            self._tp_sum_group._all_reduce_out_place = self._tp_sum_original
        else:
            del self._tp_sum_group._all_reduce_out_place
        self._tp_sum_group = None
        self.tp_reduction = "original"

    def shutdown(self):
        """Restore the TP group's original reduction and stop the worker."""
        if self.graph is not None:
            self.graph.close()
        self._restore_tp_sum()
        self.worker.shutdown()

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
        attention_names = []
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
                attention_names.append(name)
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
        if self.attention_kernel == "gather_fia":
            from .cann_attention import install_gather_fia

            context = self.config.compilation_config.static_forward_context
            for name in attention_names:
                install_gather_fia(context[name].impl)
        self.builders = [
            (names, kind(spec, names, self.config, self.device))
            for (kind, spec), names in groups.items()
        ]
        self.cache_names = sorted(caches)

    def _allocate_inputs(self):
        import torch

        self.graph = None
        self._input_buffers = {}
        for width in (1, 2, 4):
            tokens = torch.empty(width, dtype=torch.int32, device=self.device)
            positions = torch.empty(
                width, dtype=torch.long, device=self.device,
            )
            slots = torch.empty(width, dtype=torch.int32, device=self.device)
            query_cpu = torch.tensor([0, width], dtype=torch.int32)
            self._input_buffers[width] = (
                tokens, positions, slots, query_cpu, query_cpu.to(self.device),
            )
        (self.input_ids, self.positions, self.slots,
         self.query_cpu, self.query) = self._input_buffers[1]
        self.seq_lens = torch.empty(1, dtype=torch.int32, device=self.device)
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

    def _metadata(self, width=1):
        import torch
        from vllm_ascend.attention.attention_v1 import AscendAttentionState
        from vllm_ascend.attention.utils import AscendCommonAttentionMetadata

        (self.input_ids, self.positions, self.slots,
         self.query_cpu, self.query) = self._input_buffers[width]
        length_cpu = torch.tensor([self.position + width], dtype=torch.int32)
        computed_cpu = torch.tensor([self.position], dtype=torch.int32)
        positions_cpu = torch.arange(
            self.position, self.position + width, dtype=torch.long,
        )
        self.seq_lens.copy_(length_cpu)
        if width == 1:
            self.slots.fill_(self.position)
            self.positions.fill_(self.position)
        else:
            self.slots.copy_(positions_cpu)
            self.positions.copy_(positions_cpu)
        state = (AscendAttentionState.DecodeOnly if width == 1
                 else AscendAttentionState.ChunkedPrefill)
        common = AscendCommonAttentionMetadata(
            query_start_loc=self.query, query_start_loc_cpu=self.query_cpu,
            seq_lens=self.seq_lens, seq_lens_cpu=length_cpu,
            seq_lens_cpu_upper_bound=length_cpu,
            num_computed_tokens_cpu=computed_cpu,
            _seq_lens_cpu=length_cpu,
            _num_computed_tokens_cpu=computed_cpu,
            num_reqs=1, num_actual_tokens=width, max_query_len=width,
            max_seq_len=self.position + width,
            block_table_tensor=self.block_table, slot_mapping=self.slots,
            positions=self.positions, positions_cpu=positions_cpu,
            causal=True,
            is_prefilling=torch.tensor([False]),
            decode_token_per_req=1, actual_seq_lengths_q=[width],
            attn_state=state, num_input_tokens=width,
        )
        result = {}
        for names, builder in self.builders:
            metadata = builder.build(0, common)
            result.update((name, metadata) for name in names)
        return result

    def enable_graph(self, width=1):
        """Capture one query width; other widths retain eager execution."""
        from .cann_graph import CannGraph

        if self.graph is None:
            self.graph = CannGraph(self, width)
        elif self.graph.width != width:
            raise ValueError("The model already captures another width")

    def verify(self, tokens, *, full_logits=True):
        """Consume M causal rows and return M next-token predictions."""
        import torch
        from vllm.config import set_current_vllm_config
        from vllm_ascend.ascend_forward_context import (
            set_ascend_forward_context,
        )

        width = len(tokens)
        if (width not in (1, 2, 4)
                or any(type(token) is not int or not 0 <= token < 154880
                       for token in tokens)):
            raise ValueError("Verify requires 1, 2 or 4 vocabulary tokens")
        if self.position + width > self.capacity:
            raise ValueError("Decode exceeds cache capacity")
        if self.attention_kernel == "gather_fia" and self.position < 2047:
            raise ValueError("Gather/FIA requires 2048 valid selected rows")
        use_graph = self.graph is not None and self.graph.width == width
        with torch.npu.device(self.device), torch.inference_mode(), (
            set_current_vllm_config(self.config)
        ):
            metadata = self._metadata(width)
            if width == 1:
                self.input_ids.fill_(tokens[0])
            else:
                self.input_ids.copy_(torch.tensor(tokens, dtype=torch.int32))
            with set_ascend_forward_context(
                metadata, self.config, num_tokens=width,
                num_actual_tokens=width,
                model_instance=self.model,
            ):
                hidden = (self.model(self.input_ids, self.positions)
                          if not use_graph
                          else self.graph.forward(metadata))
                logits = self.model.compute_logits(hidden)
                token_ids = logits.argmax(dim=-1)
            self.position += width
            return CannStep(
                logits if full_logits else None, hidden, token_ids,
            )

    def step(self, token):
        return self.verify((token,), full_logits=False)

    def predict(self, normalized_parent, *, full_logits=False):
        """Project transferred hidden[P-1] without changing the KV length."""
        import torch
        from vllm.config import set_current_vllm_config

        if (normalized_parent.shape != (1, 6144)
                or normalized_parent.dtype != torch.bfloat16
                or normalized_parent.device != self.device):
            raise ValueError("Parent must be normalized device BF16 [1,6144]")
        with torch.npu.device(self.device), torch.inference_mode(), (
            set_current_vllm_config(self.config)
        ):
            logits = self.model.compute_logits(normalized_parent)
            return CannStep(logits if full_logits else None, normalized_parent,
                            logits.argmax(dim=-1))

    def truncate(self, length):
        """Retain written rows [0, length); later writes replace the tail."""
        if type(length) is not int or not 0 <= length <= self.position:
            raise ValueError("Truncation requires an already written prefix")
        self.position = length

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
        if self.attention_kernel == "gather_fia":
            from .cann_attention import gathered_fia

            objects.append(gathered_fia)
        if self.fp32_head:
            from .cann_head import fp32_projection

            objects.append(fp32_projection)
        if not self.config.model_config.enforce_eager:
            from vllm.compilation.decorators import support_torch_compile
            from vllm.compilation.wrapper import (
                TorchCompileWithNoGuardsWrapper,
            )
            from vllm_ascend.compilation.compiler_interface import (
                AscendCompiler,
            )

            objects.extend((AscendCompiler, support_torch_compile,
                            TorchCompileWithNoGuardsWrapper))
        if self.graph is not None:
            objects.append(type(self.graph))
        paths = {Path(inspect.getfile(inspect.unwrap(value))).resolve()
                 for value in objects}
        return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(paths)}
