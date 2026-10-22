"""Compare the fused and split arms of one dependency level, then time them.

Both arms compute the same products into the same concatenated layout and
differ only in how many matmul invocations each core issues: one over the whole
output, or one a projection. So they must agree byte for byte, and the fused arm
must be faster by the invocations it does not make.

Run with `-I` and pass the library as an argument.
"""

import argparse
import ctypes
import json
from pathlib import Path

import torch
import torch_npu

SYSTEM_BYTES = 16 << 20

# Each level's shared activation width and the projections that read it, per
# rank at TP16.
LEVELS = {
    'hidden': {
        'reduction': 6144,
        'parts': {'query_latent': 2048, 'kv_latent_rope': 576,
                  'indexer_key': 160, 'router_gate': 256},
    },
    'query_norm': {
        'reduction': 2048,
        'parts': {'query_up': 1024, 'indexer_query': 4096},
    },
}


def load(library):
    ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    return torch.ops.megakernel_front


def operands(level, rows, stride, device, generator):
    return {
        'activation': torch.randn(rows, level['reduction'],
                                  dtype=torch.bfloat16, device=device,
                                  generator=generator),
        'weight': torch.randn(level['reduction'], stride,
                              dtype=torch.bfloat16, device=device,
                              generator=generator),
    }


def submit(front, held, parts, blocks, sync, tiling, system, product):
    front.forward(held['activation'], held['weight'], product, tiling, system,
                  blocks, parts, sync)


def timed(submit_once, iterations):
    for _ in range(5):
        submit_once()
    torch.npu.synchronize()
    start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(iterations):
        submit_once()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) * 1e3 / iterations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('library', type=Path)
    parser.add_argument('--level', choices=sorted(LEVELS), default='hidden')
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--rows', type=int, default=12)
    parser.add_argument('--blocks', type=int, default=24)
    parser.add_argument('--inputs', type=int, default=8)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--stride', type=int, default=0)
    parser.add_argument('--base-m', type=int, default=-1)
    parser.add_argument('--base-n', type=int, default=-1)
    parser.add_argument('--base-k', type=int, default=-1)
    args = parser.parse_args()
    torch.npu.set_device(args.device)

    front = load(args.library)
    level = LEVELS[args.level]
    parts = list(level['parts'].values())
    columns = sum(parts)
    stride = args.stride or front.aligned_width(columns)
    sync = front.sync_base()
    tiling = front.build_tiling(args.rows, args.blocks, level['reduction'],
                                parts, stride, args.base_m, args.base_n,
                                args.base_k).to(args.device)
    system = torch.zeros(SYSTEM_BYTES, dtype=torch.uint8, device=args.device)

    generator = torch.Generator(device=args.device).manual_seed(20261022)
    cases = []
    for index in range(args.inputs):
        held = operands(level, args.rows, stride, args.device, generator)
        product = torch.zeros(args.rows, stride, dtype=torch.float,
                              device=args.device)
        submit(front, held, parts, args.blocks, sync, tiling, system, product)
        torch.npu.synchronize()
        computed = product[:, :columns]
        reference = (held['activation'].float()
                     @ held['weight'][:, :columns].float())
        cases.append({
            'input': index,
            'product_sha256': torch.sum(
                computed.cpu().contiguous().view(torch.int32).long()).item(),
            'max_absolute_difference': (computed - reference).abs().max().item(),
            'reference_magnitude': reference.abs().max().item(),
            'finite': bool(torch.isfinite(computed).all()),
        })

    held = operands(level, args.rows, stride, args.device, generator)
    product = torch.zeros(args.rows, stride, dtype=torch.float,
                          device=args.device)
    samples = [timed(lambda: submit(front, held, parts, args.blocks, sync,
                                    tiling, system, product), args.iterations)
               for _ in range(args.repeats)]

    print(json.dumps({
        'library': str(args.library), 'level': args.level,
        'reduction': level['reduction'], 'parts': level['parts'],
        'columns': columns, 'weight_stride': stride,
        'rows': args.rows, 'blocks': args.blocks,
        'base': [args.base_m, args.base_n, args.base_k],
        'cases': cases,
        'median_us': round(sorted(samples)[len(samples) // 2], 2),
        'samples_us': [round(value, 2) for value in samples],
    }, indent=2))


if __name__ == '__main__':
    main()
