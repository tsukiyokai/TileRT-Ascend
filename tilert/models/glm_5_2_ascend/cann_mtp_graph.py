"""Fixed M1 graph for GLM-5.2's shared-index MTP draft chain."""


class CannMtpGraph:
    """Replay the draft and its own head; returned storage is reused.

    Warm the eager M1 path on every TP rank before enabling this graph.
    Seed and target-conditioned forwards retain eager execution.
    """

    def __init__(self, draft):
        import torch

        self.draft = draft
        self.tokens, self.positions, self.slots = draft._input_buffers[1][:3]
        self.parent = torch.empty(
            1, 6144, dtype=torch.bfloat16, device=draft.device,
        )
        self.slots_i64 = torch.empty(
            1, dtype=torch.int64, device=draft.device,
        )
        self.graph = torch.npu.NPUGraph()
        self.pool = torch.npu.graph_pool_handle()
        self.capture_count = 0
        self.replay_count = 0
        self.output = None
        self.metadata = None
        self.addresses = None
        self.closed = False

    @staticmethod
    def _addresses(metadata):
        from .cann_graph import CannGraph

        return {
            name: {field: CannGraph._signature(getattr(value, field))
                   for field in CannGraph._METADATA_FIELDS}
            for name, value in metadata.items()
        }

    def validate_metadata(self, metadata):
        """Check persistent replay buffers outside the measured path."""
        if self._addresses(metadata) != self.addresses:
            raise ValueError("MTP graph metadata addresses changed")

    def _execute(self):
        model = self.draft.model
        pre_norm, recycle = model(
            input_ids=self.tokens, positions=self.positions,
            hidden_states=self.parent, spec_step_idx=0,
        )
        logits = model.compute_logits(pre_norm, spec_step_idx=0)
        return (pre_norm, recycle, logits, logits.argmax(dim=-1),
                self.draft.indices_buffer[:1].clone())

    def forward(self, metadata, parent):
        import torch
        from vllm.config import set_current_vllm_config
        from vllm_ascend.ascend_forward_context import (
            set_ascend_forward_context,
        )
        from vllm_ascend.attention.sfa_v1 import AscendSFAMetadata
        from vllm_ascend.worker.model_runner_v1 import graph_capture

        if self.closed:
            raise RuntimeError("MTP graph is closed")
        # A chained parent can be this graph's previous recycle output.
        self.parent.copy_(parent)
        self.slots_i64.copy_(self.slots)
        self.draft.model.model.set_skip_topk(True)
        if not self.capture_count:
            for value in metadata.values():
                if isinstance(value, AscendSFAMetadata):
                    value.kv_slots_i64 = (value.slot_mapping, self.slots_i64)
            self.metadata = metadata
            self.addresses = self._addresses(metadata)
        with (
            set_current_vllm_config(self.draft.config),
            set_ascend_forward_context(
                metadata, self.draft.config, num_tokens=1,
                num_actual_tokens=1, model_instance=self.draft.model,
                is_draft_model=True,
            ),
        ):
            if not self.capture_count:
                caller = torch.npu.current_stream(self.draft.device)
                with graph_capture(self.draft.device) as capture:
                    with torch.npu.graph(
                        self.graph, pool=self.pool, stream=capture.stream,
                    ):
                        self.output = self._execute()
                caller.wait_stream(capture.stream)
                self.capture_count += 1
            self.graph.replay()
            self.replay_count += 1
        return self.output

    def close(self):
        import torch

        if not self.closed:
            with torch.npu.device(self.draft.device):
                torch.npu.synchronize(self.draft.device)
                self.graph.reset()
            self.metadata = self.output = None
            self.closed = True
