"""CPU HF decoder with independent ModelSlim and TP16 arithmetic adapters."""

import json
import math
from types import FunctionType, MethodType

import torch
from torch import nn
import torch.nn.functional as functional
from torch.overrides import TorchFunctionMode
from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
    GlmMoeDsaRotaryEmbedding,
)

from .reference_weights import ReferenceWeights, quantize_a8, tp_sum
from .ops.indexer import (
    CANN_ARCH22_SOURCE_DIRECTORY,
    cann_arch22_index_scores,
    cann_arch22_input_domain,
    cann_arch22_scoring_contract,
)


TP_SIZE = 16
LAYERS = 78
ROTARY_CONTRACT = {
    "intermediates": "FP32 products and sum; one BF16 output cast",
    "angles": "HF FP32 generation followed by BF16 cast",
    "main_basis": "real half, imaginary half",
    "index_basis": "alternating real and imaginary",
    "cann_source_sha256": (
        "96028ea9c3f2b5426b60c00e03951fafbc"
        "d4903ceff5880b4ad31482d37e0a8a"
    ),
    "hf_adapter": "Instance forward globals; HF bytecode unchanged",
}


class StableTopk(TorchFunctionMode):
    """Resolve equal router scores by ascending expert ID."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func not in (torch.topk, torch.Tensor.topk):
            return func(*args, **kwargs)
        scores = args[0] if args else kwargs["input"]
        count = args[1] if len(args) > 1 else kwargs["k"]
        dimension = kwargs.get("dim", args[2] if len(args) > 2 else -1)
        largest = kwargs.get("largest", args[3] if len(args) > 3 else True)
        indices = scores.argsort(
            dim=dimension, descending=largest, stable=True
        ).narrow(dimension, 0, count)
        return torch.return_types.topk((
            scores.gather(dimension, indices), indices,
        ))


class ReferenceLinear(nn.Module):
    def __init__(self, weights, prefix, partition="replicated"):
        super().__init__()
        self.weights = weights
        self.prefix = prefix
        self.partition = partition

    def forward(self, x):
        return self.weights.linear(self.prefix, x, partition=self.partition)


class ReferenceEmbedding(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def forward(self, tokens):
        if tokens.shape != (1, 1):
            raise ValueError("The reference consumes one token at a time")
        token = int(tokens.item())
        row = self.weights.tensor(
            "model.embed_tokens.weight", rows=(token, token + 1),
            dtype=torch.bfloat16,
        )
        return row.unsqueeze(0)


def fused_swiglu(gate, up):
    """Round the FP32 activation/product once at the BF16 FFN boundary."""
    return (functional.silu(gate.float()) * up.float()).to(torch.bfloat16)


def ffn_partials(weights, prefix, hidden):
    gate = weights.linear(prefix + ".gate_proj", hidden, partition="output")
    up = weights.linear(prefix + ".up_proj", hidden, partition="output")
    activated = fused_swiglu(gate, up)
    return weights.input_partials(prefix + ".down_proj", activated)


class ReferenceMlp(nn.Module):
    def __init__(self, weights, prefix):
        super().__init__()
        self.weights = weights
        self.prefix = prefix

    def forward(self, hidden):
        return tp_sum(ffn_partials(self.weights, self.prefix, hidden))


class ReferenceMoe(nn.Module):
    """Retain HF routing and combine experts before each rank's TP sum."""

    def __init__(self, weights, prefix, gate):
        super().__init__()
        self.weights = weights
        self.prefix = prefix
        self.gate = gate

    def forward(self, hidden):
        with StableTopk():
            _, probabilities, choices = self.gate(hidden)
        probabilities = probabilities[0].to(torch.bfloat16)
        flat = hidden.reshape(1, -1)
        expert_partials = [
            ffn_partials(self.weights, f"{self.prefix}.experts.{expert}", flat)
            for expert in choices[0].tolist()
        ]
        shared = ffn_partials(
            self.weights, self.prefix + ".shared_experts", flat
        )
        rank_outputs = []
        for rank in range(TP_SIZE):
            products = [
                values[rank].float() * probability.float()
                for values, probability in zip(expert_partials, probabilities)
            ]
            combined = torch.stack(products).sum(dim=0).to(torch.bfloat16)
            rank_outputs.append(combined + shared[rank])
        return tp_sum(rank_outputs).reshape_as(hidden)


def rotary_components(x, cosine, sine, head_axis):
    """Apply CANN's FP32 rotary intermediates to BF16 inputs and angles."""
    half = x.shape[-1] // 2
    cosine = cosine[..., :half].float().unsqueeze(head_axis)
    sine = sine[..., :half].float().unsqueeze(head_axis)
    even, odd = x[..., 0::2].float(), x[..., 1::2].float()
    return even * cosine + (-odd) * sine, odd * cosine + even * sine


def interleaved_rope(x, cosine, sine):
    """Round once to BF16 and retain alternating indexer coordinates."""
    first, second = rotary_components(x, cosine, sine, head_axis=2)
    return torch.stack((first, second), dim=-1).flatten(-2).to(x.dtype)


def attention_rope(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Retain HF's half-basis layout with CANN's rotary rounding boundary."""
    query = rotary_components(q, cos, sin, unsqueeze_dim)
    key = rotary_components(k, cos, sin, unsqueeze_dim)
    return (torch.cat(query, dim=-1).to(q.dtype),
            torch.cat(key, dim=-1).to(k.dtype))


def install_attention_rope(attention):
    """Change only this instance's HF forward rotary function lookup."""
    original = attention.forward.__func__
    namespace = dict(original.__globals__)
    namespace["apply_rotary_pos_emb_interleave"] = attention_rope
    adapted = FunctionType(original.__code__, namespace, original.__name__,
                           original.__defaults__, original.__closure__)
    adapted.__kwdefaults__ = original.__kwdefaults__
    attention.forward = MethodType(adapted, attention)


def hadamard_matrix():
    signs = [
        [1 if (row & column).bit_count() % 2 == 0 else -1
         for column in range(128)]
        for row in range(128)
    ]
    return torch.tensor(signs, dtype=torch.bfloat16) / math.sqrt(128)


def prefix_length(prefixes):
    """Validate the replicated backbone cache before changing HF state."""
    if len(prefixes) != LAYERS:
        raise ValueError("The reference needs 78 backbone prefixes")
    length = prefixes[0]["latent"].shape[0]
    for layer, entry in enumerate(prefixes):
        expected = {
            "latent": ((length, 512), torch.bfloat16),
            "rope": ((length, 64), torch.bfloat16),
        }
        full = layer < 3 or (layer - 2) % 4 == 0
        if full:
            quantized = 6 <= layer <= 66
            dtype = torch.int8 if quantized else torch.bfloat16
            expected["index_keys"] = (length, 128), dtype
            if quantized:
                expected["index_scales"] = (length,), torch.float16
        if entry.keys() != expected.keys():
            raise ValueError(f"Wrong prefix regions at layer {layer}")
        for name, (shape, dtype) in expected.items():
            value = entry[name]
            if (tuple(value.shape) != shape or value.dtype != dtype
                    or value.device.type != "cpu"):
                raise ValueError(f"Invalid prefix {layer}.{name}")
    return length


class ReferenceIndexer(nn.Module):
    """Independent CPU scoring over BF16 or C8 keys in HF DynamicCache."""

    def __init__(self, weights, prefix, layer, quantized):
        super().__init__()
        self.weights = weights
        self.prefix = prefix
        self.layer_idx = layer
        self.quantized = quantized
        self.rotation = hadamard_matrix() if quantized else None
        self.scales = None
        self.last_key = None
        self.last_scale = None

    def reset(self):
        self.scales = None
        self.last_key = None
        self.last_scale = None

    def encode(self, x):
        if not self.quantized:
            return x, None
        encoded, scale = quantize_a8(x @ self.rotation)
        return encoded, scale.to(torch.float16)

    def inject(self, cache, keys, scales):
        expected = torch.int8 if self.quantized else torch.bfloat16
        if keys.dtype != expected or keys.ndim != 2 or keys.shape[1] != 128:
            raise ValueError("Invalid encoded index prefix")
        cache.update_indexer(keys.unsqueeze(0), self.layer_idx)
        if self.quantized:
            if (scales is None or scales.dtype != torch.float16
                    or scales.numel() != keys.shape[0]):
                raise ValueError("C8 prefix needs one FP16 scale per key")
            self.scales = scales.reshape(-1).clone()

    def forward(
        self, hidden_states, q_resid, position_embeddings, attention_mask,
        position_ids, past_key_values=None,
    ):
        if hidden_states.shape[:2] != (1, 1) or past_key_values is None:
            raise ValueError("The reference indexer requires cached M=1")
        query = self.weights.linear(
            self.prefix + ".wq_b", q_resid, partition="replicated"
        )
        query = query.reshape(1, 1, 32, 128)
        key = self.weights.linear(
            self.prefix + ".wk", hidden_states, partition="replicated"
        )
        gamma = self.weights.tensor(self.prefix + ".k_norm.weight")
        beta = self.weights.tensor(self.prefix + ".k_norm.bias")
        key = functional.layer_norm(
            key.float(), (128,), gamma.float(), beta.float(), eps=1e-6
        ).to(torch.bfloat16).unsqueeze(2)
        cosine, sine = position_embeddings
        query = torch.cat((
            interleaved_rope(query[..., :64], cosine, sine), query[..., 64:]
        ), dim=-1).reshape(32, 128)
        key = torch.cat((
            interleaved_rope(key[..., :64], cosine, sine), key[..., 64:]
        ), dim=-1).reshape(1, 128)
        encoded_key, key_scale = self.encode(key)
        keys = past_key_values.update_indexer(
            encoded_key.unsqueeze(0), self.layer_idx
        )[0]
        self.last_key = encoded_key[0].clone()
        self.last_scale = None if key_scale is None else key_scale.reshape(1)
        query, query_scale = self.encode(query)
        weights = self.weights.linear(
            self.prefix + ".weights_proj", hidden_states,
            partition="replicated",
        ).reshape(32)
        if self.quantized:
            scale = key_scale.reshape(-1)
            self.scales = (scale if self.scales is None else
                           torch.cat((self.scales, scale)))
            rounded_weights = weights.to(torch.float16)
            domain = cann_arch22_input_domain(
                query_scale.reshape(32).tolist(), self.scales.tolist(),
                rounded_weights.tolist(),
            )
            if not domain["accepted"]:
                raise ValueError(
                    "Unverified CANN FP16 scoring boundary: "
                    + json.dumps(domain, sort_keys=True)
                )
            scores = cann_arch22_index_scores(
                query, keys, query_scale.reshape(32), self.scales,
                rounded_weights,
            )
        else:
            scores = query.float() @ keys.float().T
            scores = (scores.relu() * weights.float()[:, None]).sum(dim=0)
        if attention_mask is not None:
            scores = scores + attention_mask.reshape(-1).float()
        else:
            scores = scores.masked_fill(
                torch.arange(keys.shape[0]) > position_ids.item(), -torch.inf
            )
        indices = scores.argsort(descending=True, stable=True)[:2048]
        return indices.to(torch.int32).reshape(1, 1, -1)


class HfReference:
    """Run the installed HF model without CANN or production weight helpers."""

    def __init__(self, checkpoint):
        self.weights = ReferenceWeights(checkpoint)
        self.weights.contract["rotary"] = dict(ROTARY_CONTRACT)
        self.weights.contract["c8_index_scoring"] = (
            cann_arch22_scoring_contract(CANN_ARCH22_SOURCE_DIRECTORY)
        )
        raw = json.loads((self.weights.directory / "config.json").read_text())
        config = GlmMoeDsaConfig.from_dict(raw)
        config._attn_implementation = "eager"
        config._experts_implementation = "eager"
        if (config.num_hidden_layers != LAYERS or config.n_group != 1
                or config.topk_group != 1 or config.n_shared_experts != 1
                or config.rms_norm_eps != 1e-5):
            raise ValueError("The reference requires the GLM-5.2 TP16 layout")
        with torch.device("meta"):
            self.model = GlmMoeDsaForCausalLM(config).to(torch.bfloat16)
        self.config = config
        self._install_adapters()
        self.model.model.rotary_emb = GlmMoeDsaRotaryEmbedding(
            config=config, device="cpu"
        )
        self.model.eval().requires_grad_(False)
        remaining = [name for name, value in (
            *self.model.named_parameters(), *self.model.named_buffers()
        ) if value.is_meta]
        if remaining:
            raise ValueError(f"Unreplaced HF meta tensors: {remaining}")
        self._attach_trace_hooks()
        self.reset()

    def _norm(self, module, prefix):
        module.weight = nn.Parameter(self.weights.tensor(
            prefix + ".weight", dtype=torch.bfloat16
        ), requires_grad=False)
        module.variance_epsilon = 1e-5

    def _install_adapters(self):
        self.model.model.embed_tokens = ReferenceEmbedding(self.weights)
        self.model.lm_head = ReferenceLinear(
            self.weights, "lm_head", partition="output"
        )
        self._norm(self.model.model.norm, "model.norm")
        for number, layer in enumerate(self.model.model.layers):
            install_attention_rope(layer.self_attn)
            prefix = f"model.layers.{number}"
            attention = prefix + ".self_attn"
            self._norm(layer.input_layernorm, prefix + ".input_layernorm")
            self._norm(layer.post_attention_layernorm,
                       prefix + ".post_attention_layernorm")
            self._norm(layer.self_attn.q_a_layernorm,
                       attention + ".q_a_layernorm")
            self._norm(layer.self_attn.kv_a_layernorm,
                       attention + ".kv_a_layernorm")
            for name, partition in (
                ("q_a_proj", "replicated"), ("q_b_proj", "output"),
                ("kv_a_proj_with_mqa", "replicated"),
                ("kv_b_proj", "output"), ("o_proj", "input"),
            ):
                setattr(layer.self_attn, name, ReferenceLinear(
                    self.weights, attention + "." + name, partition
                ))
            if layer.self_attn.indexer is not None:
                index = attention + ".indexer"
                recipe = self.weights.quantization.get(index + ".quant_type")
                layer.self_attn.indexer = ReferenceIndexer(
                    self.weights, index, number, recipe == "INT8_DYNAMIC"
                )
            mlp = prefix + ".mlp"
            if hasattr(layer.mlp, "gate"):
                gate = layer.mlp.gate
                gate.weight = nn.Parameter(self.weights.tensor(
                    mlp + ".gate.weight", dtype=torch.float32
                ), requires_grad=False)
                gate.e_score_correction_bias = self.weights.tensor(
                    mlp + ".gate.e_score_correction_bias", dtype=torch.float32
                )
                layer.mlp = ReferenceMoe(self.weights, mlp, gate)
            else:
                layer.mlp = ReferenceMlp(self.weights, mlp)

    def _attach_trace_hooks(self):
        for number, layer in enumerate(self.model.model.layers):
            def record_attention(module, inputs, output, index=number):
                self.indices[index] = output[2][0, -1].detach().clone()

            def record_latent(module, inputs, output, index=number):
                self.latents[index] = output[0, -1].detach().clone()

            def record_route(module, inputs, output, index=number):
                self.routes[index] = output[2][0].detach().clone()

            layer.self_attn.register_forward_hook(record_attention)
            layer.self_attn.kv_a_layernorm.register_forward_hook(record_latent)
            if hasattr(layer.mlp, "gate"):
                layer.mlp.gate.register_forward_hook(record_route)

    def reset(self):
        self.cache = DynamicCache(config=self.config)
        for layer in self.model.model.layers:
            if layer.self_attn.indexer is not None:
                layer.self_attn.indexer.reset()

    @property
    def position(self):
        return self.cache.get_seq_length()

    def inject(self, prefixes):
        prefix_length(prefixes)
        if self.position:
            raise ValueError("Inject prefixes into an empty reference")
        for number, (layer, entry) in enumerate(zip(
            self.model.model.layers, prefixes
        )):
            latent, rope = entry["latent"], entry["rope"]
            expanded = layer.self_attn.kv_b_proj(latent)
            expanded = expanded.reshape(-1, 64, 448)
            key, value = expanded.split((192, 256), dim=-1)
            rope = rope[:, None, :].expand(-1, 64, 64)
            key = torch.cat((key, rope), dim=-1).transpose(0, 1).unsqueeze(0)
            value = value.transpose(0, 1).unsqueeze(0)
            self.cache.update(key.contiguous(), value.contiguous(), number)
            if layer.self_attn.indexer is not None:
                layer.self_attn.indexer.inject(
                    self.cache, entry["index_keys"], entry.get("index_scales")
                )

    @torch.inference_mode()
    def step(self, token):
        self.indices, self.routes, self.latents = (
            [None] * LAYERS for _ in range(3)
        )
        position = self.position
        output = self.model(
            input_ids=torch.tensor([[token]]), past_key_values=self.cache,
            use_cache=True, output_hidden_states=True, return_dict=True,
        )
        self.cache = output.past_key_values
        hidden = tuple(value[:, -1].clone() for value in output.hidden_states)
        normalized = self.model.model.norm(torch.cat(hidden[1:-1]))
        projected = self.model.lm_head(normalized)
        logits = output.logits[:, -1].clone()
        layer_logits = (*projected.split(1, dim=0), logits)
        rows = []
        for number, layer in enumerate(self.model.model.layers):
            indexer = layer.self_attn.indexer
            rows.append({
                "latent": self.latents[number],
                "rope": self.cache.layers[number].keys[0, 0, -1, 192:].clone(),
                "index": None if indexer is None else indexer.last_key.clone(),
                "index_scale": (
                    None if indexer is None or not indexer.quantized
                    else indexer.last_scale.clone()
                ),
            })
        return {
            "token": token, "position": position,
            "committed_position": self.position, "logits": logits,
            "hidden_states": hidden, "layer_logits": layer_logits,
            "indices": tuple(self.indices), "routes": tuple(self.routes),
            "cache_rows": tuple(rows),
        }
