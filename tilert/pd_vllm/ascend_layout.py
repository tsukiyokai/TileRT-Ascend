"""TP16 GLM-5.2 KV layout and page-copy plans for HIXL WRITE."""

from operator import index

PAGE_SIZE = 128
NUM_RANKS = 16
NUM_LAYERS = 79
FULL_INDEX_LAYERS = (0, 1, 2, *range(6, 79, 4))
INT8_INDEX_LAYERS = frozenset(range(6, 67, 4))
LAYOUT_NAME = "glm5_2_ascend_w4a8c8"
LAYOUT_VERSION = 2


def validate_prompt_token_ids(tokens, seq_len, last_prompt_token):
    """Bind the complete vocabulary IDs to the received prefix length."""
    if (
        type(seq_len) is not int or seq_len < 1
        or not isinstance(tokens, list) or len(tokens) != seq_len
        or any(type(token) is not int or not 0 <= token < 154880
               for token in tokens)
    ):
        raise ValueError("Prompt IDs must match the received prefix length")
    if (
        type(last_prompt_token) is not int
        or tokens[-1] != last_prompt_token
    ):
        raise ValueError("Prompt IDs differ from the final prompt token")


def cache_layout(capacity_tokens=9344):
    """Describe one rank's contiguous receive buffer, including MTP layer 78.

    BF16 latent[512], RoPE[64], index keys, and FP16 index scales occupy
    separate regions. Shared indexer layers own no region.
    Both peers use 128-token pages and the checkpoint's mixed index recipe.
    """
    capacity_tokens = index(capacity_tokens)
    if capacity_tokens < 1:
        raise ValueError("cache capacity must be positive")
    pages = (capacity_tokens + PAGE_SIZE - 1) // PAGE_SIZE
    padded_capacity = pages * PAGE_SIZE
    regions = {}
    offset = 0

    def add(name, dtype, row_bytes, shape):
        nonlocal offset
        nbytes = padded_capacity * row_bytes
        regions[name] = {
            "dtype": dtype, "row_bytes": row_bytes, "shape": shape,
            "offset": offset, "nbytes": nbytes,
        }
        offset += nbytes

    for layer in range(NUM_LAYERS):
        add(
            f"mla_latent.{layer}", "bfloat16", 1024,
            (pages, PAGE_SIZE, 1, 512),
        )
        add(
            f"mla_rope.{layer}", "bfloat16", 128,
            (pages, PAGE_SIZE, 1, 64),
        )
    for layer in FULL_INDEX_LAYERS:
        quantized = layer in INT8_INDEX_LAYERS
        add(
            f"index.{layer}", "int8" if quantized else "bfloat16",
            128 if quantized else 256, (pages, PAGE_SIZE, 1, 128),
        )
        if quantized:
            add(f"index_scale.{layer}", "float16", 2, (pages, PAGE_SIZE, 1))
    return {
        "name": LAYOUT_NAME, "version": LAYOUT_VERSION, "num_ranks": NUM_RANKS,
        "page_size": PAGE_SIZE, "capacity_tokens": padded_capacity,
        "nbytes": offset, "regions": regions,
    }


def validate_layout(layout):
    """Require the complete split-region contract after JSON serialization."""
    expected = cache_layout(layout["capacity_tokens"])
    for field in (
        "name", "version", "num_ranks", "page_size", "capacity_tokens",
        "nbytes",
    ):
        if layout[field] != expected[field]:
            raise ValueError(f"Ascend KV layout mismatch: {field}")
    if set(layout["regions"]) != set(expected["regions"]):
        raise ValueError("Ascend KV layout must contain all 196 regions")
    for name, region in expected["regions"].items():
        actual = layout["regions"][name]
        for field, value in region.items():
            given = tuple(actual[field]) if field == "shape" else actual[field]
            if given != value:
                raise ValueError(f"Ascend KV layout mismatch: {name}.{field}")


def page_transfer_plan(
    layout, source_ptrs, source_blocks, destination_ptr, prefix_tokens, *,
    source_page_strides=None, source_block_counts=None, mtp_prefix_tokens=None,
):
    """Return (sources, destinations, lengths) for one rank's HIXL WRITE.

    source_ptrs and source_blocks are keyed by region name, so distinct vLLM
    KV cache groups may have different physical block tables. Destination
    pages are contiguous from zero for the admitted batch-one request.
    Adjacent source pages coalesce within each registered source tensor;
    the final page copies only valid tokens. A vLLM SourceBinding supplies
    physical page strides and block counts from the actual tensor views.
    Omitted strides describe independent, contiguous source regions.
    MTP may omit the final row, whose input contains the prefill first token.
    """
    validate_layout(layout)
    prefix_tokens = index(prefix_tokens)
    destination_ptr = index(destination_ptr)
    if destination_ptr <= 0:
        raise ValueError("destination pointer must be positive")
    if not 0 < prefix_tokens <= layout["capacity_tokens"]:
        raise ValueError("prefix exceeds the receive buffer capacity")
    mtp_prefix_tokens = (
        prefix_tokens if mtp_prefix_tokens is None
        else index(mtp_prefix_tokens)
    )
    if mtp_prefix_tokens not in (prefix_tokens - 1, prefix_tokens):
        raise ValueError("MTP prefix must contain P-1 or P rows")
    sources, destinations, lengths = [], [], []
    for name, region in layout["regions"].items():
        valid_tokens = (
            mtp_prefix_tokens if name.endswith(".78") else prefix_tokens
        )
        if valid_tokens == 0:
            continue
        required_pages = (valid_tokens + PAGE_SIZE - 1) // PAGE_SIZE
        blocks = source_blocks[name]
        if len(blocks) < required_pages:
            raise ValueError(f"{name}: incomplete source block table")
        blocks = [index(block) for block in blocks[:required_pages]]
        if any(block < 0 for block in blocks):
            raise ValueError(f"{name}: negative source block")
        if source_block_counts is not None:
            if max(blocks) >= source_block_counts[name]:
                raise ValueError(f"{name}: source block exceeds tensor")
        page_bytes = PAGE_SIZE * region["row_bytes"]
        stride = (
            page_bytes if source_page_strides is None
            else index(source_page_strides[name])
        )
        if stride < page_bytes:
            raise ValueError(f"{name}: source pages overlap")
        pointer = index(source_ptrs[name])
        if pointer <= 0:
            raise ValueError(f"{name}: source pointer must be positive")
        run = None
        for page in range(required_pages):
            count = min(PAGE_SIZE, valid_tokens - page * PAGE_SIZE)
            source = pointer + blocks[page] * stride
            destination = (
                destination_ptr + region["offset"] + page * page_bytes
            )
            nbytes = count * region["row_bytes"]
            if (
                run is not None and sources[run] + lengths[run] == source
                and destinations[run] + lengths[run] == destination
            ):
                lengths[run] += nbytes
            else:
                sources.append(source)
                destinations.append(destination)
                lengths.append(nbytes)
                run = len(sources) - 1
    return sources, destinations, lengths
