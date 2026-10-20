"""Does the repack preserve the codes it moves?

`install` has to turn the checkpoint's packing into the kernel's. The
checkpoint pairs two INT4 codes in a byte along the output axis; the kernel
pairs them along K. Both use the same nibble order, so nothing in a dtype or
a shape would catch a wrong permutation -- only the codes themselves will.

The property: unpacking the repacked tensor along K must give the same logical
codes as unpacking the original along N. CPU only.
"""
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))

from tilert.models.glm_5_2_ascend.routed_expert import _repack


def signed(nibble):
    return torch.where(nibble > 7, nibble.int() - 16, nibble.int())


def unpack_along_outputs(packed, columns):
    """The checkpoint's reading: byte holds outputs 2n and 2n+1 for one K."""
    experts, halved, inner = packed.shape
    low = signed(packed & 0x0F)
    high = signed((packed >> 4) & 0x0F)
    stacked = torch.stack((low, high), dim=2)
    return stacked.reshape(experts, 2 * halved, inner)


def unpack_along_inner(packed):
    """The kernel's reading: byte holds K and K+1 for one output."""
    low = signed(packed & 0x0F)
    high = signed((packed >> 4) & 0x0F)
    experts, columns, halved = packed.shape
    stacked = torch.stack((low, high), dim=3)
    return stacked.reshape(experts, columns, 2 * halved)


def main():
    torch.manual_seed(20261020)
    for experts, columns, inner in ((2, 8, 16), (3, 256, 64), (1, 64, 128)):
        original = torch.randint(0, 256, (experts, columns // 2, inner),
                                 dtype=torch.uint8).to(torch.int8)
        expected = unpack_along_outputs(original, columns)
        actual = unpack_along_inner(_repack(original, columns))
        same = bool(torch.equal(actual, expected))
        print(f"{experts}x{columns}x{inner}: codes preserved {same}  "
              f"unequal {int((actual != expected).sum())}/{actual.numel()}")
        if not same:
            raise SystemExit("repack does not preserve the codes")


if __name__ == "__main__":
    main()
