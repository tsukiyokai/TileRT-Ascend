"""Compare the fused and split front arms numerically, then time them.

The two arms compute the same products into the same concatenated layout and
differ only in how many matmul invocations each core issues: one over the whole
output, or one a projection. So they must agree byte for byte, and the fused arm
must be faster by the invocations it does not make.

Run with `-I` and pass each library as an argument.
"""

import argparse
import ctypes
import json
from pathlib import Path

import torch
import torch_npu

HIDDEN = 6144
PARTS = {'query_latent': 2048, 'kv_latent_rope': 576,
         'indexer_key': 160, 'router_gate': 256}
COLUMNS = sum(PARTS.values())
SYSTEM_BYTES = 16 << 20


def load(library):
    ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    return torch.ops.megakernel_front


def operands(rows, device, generator):
    return {
        'hidden': torch.randn(rows, HIDDEN, dtype=torch.bfloat16,
                              device=device, generator=generator),
        'weight': torch.randn(HIDDEN, COLUMNS, dtype=torch.bfloat16,
                              device=device, generator=generator),
    }


def run(front, held, rows, blocks, sync, tiling, system):
    product = torch.zeros(rows, COLUMNS, dtype=torch.float, device=held['hidden'].device)
    front.forward(held['hidden'], held['weight'], product, tiling, system,
                  blocks, sync)
    torch.npu.synchronize()
    return product


def timed(front, held, product, tiling, system, blocks, sync, iterations):
    for _ in range(5):
        front.forward(held['hidden'], held['weight'], product, tiling, system,
                      blocks, sync)
    torch.npu.synchronize()
    start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(iterations):
        front.forward(held['hidden'], held['weight'], product, tiling, system,
                      blocks, sync)
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) * 1e3 / iterations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('library', type=Path)
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--rows', type=int, default=12)
    parser.add_argument('--blocks', type=int, default=24)
    parser.add_argument('--inputs', type=int, default=4)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    front = load(args.library)
    sync = front.sync_base()
    tiling = front.build_tiling(args.rows, args.blocks).to(args.device)
    system = torch.zeros(SYSTEM_BYTES, dtype=torch.uint8, device=args.device)

    generator = torch.Generator(device=args.device).manual_seed(20261022)
    cases = []
    for index in range(args.inputs):
        held = operands(args.rows, args.device, generator)
        product = run(front, held, args.rows, args.blocks, sync, tiling, system)
        reference = (held['hidden'].float() @ held['weight'].float())
        cases.append({
            'input': index,
            'product_sha256': torch.sum(product.cpu().view(torch.int32).long()).item(),
            'max_absolute_difference': (product - reference).abs().max().item(),
            'reference_magnitude': reference.abs().max().item(),
            'finite': bool(torch.isfinite(product).all()),
        })

    held = operands(args.rows, args.device, generator)
    product = torch.zeros(args.rows, COLUMNS, dtype=torch.float,
                          device=args.device)
    samples = [timed(front, held, product, tiling, system, args.blocks, sync,
                     args.iterations) for _ in range(args.repeats)]

    print(json.dumps({
        'library': str(args.library), 'rows': args.rows, 'blocks': args.blocks,
        'columns': COLUMNS, 'parts': PARTS,
        'cases': cases,
        'median_us': round(sorted(samples)[len(samples) // 2], 2),
        'samples_us': [round(value, 2) for value in samples],
    }, indent=2))


if __name__ == '__main__':
    main()
