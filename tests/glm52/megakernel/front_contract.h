#ifndef TESTS_GLM52_MEGAKERNEL_FRONT_CONTRACT_H
#define TESTS_GLM52_MEGAKERNEL_FRONT_CONTRACT_H

#include <cstdint>

// The layer's first dependency level: every projection that reads the
// normalized hidden. A decode matmul's cost follows its reduction length, paid
// once an invocation, while its output axis is spread across cores for free --
// so these four belong in one invocation over their concatenated output.
namespace megakernel_front {

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
#define MEGAKERNEL_FRONT_DEVICE __aicore__
#else
#define MEGAKERNEL_FRONT_DEVICE
#endif

constexpr int64_t kHidden = 6144;
constexpr int64_t kQueryLatent = 2048;
constexpr int64_t kKeyValueLatent = 576;   // 512 latent and 64 rope
constexpr int64_t kIndexerKey = 160;       // kDim 128 and kHeads 32
constexpr int64_t kRouterGate = 256;       // one column an expert
constexpr int64_t kParts = 4;
constexpr int64_t kColumns =
    kQueryLatent + kKeyValueLatent + kIndexerKey + kRouterGate;

constexpr int64_t kColumnGranularity = 32;
// A weight row must start on a 512-byte boundary or every core's column window
// straddles one. Padding the 3040 columns to 3072 takes the stage from 48.12 to
// 38.43 microseconds; see megakernel-front-20261022.
constexpr int64_t kStrideBytes = 512;
constexpr int64_t kAlignedColumns =
    (kColumns * 2 + kStrideBytes - 1) / kStrideBytes * kStrideBytes / 2;
constexpr int64_t kMaxRows = 32;
constexpr int64_t kMaxBlocks = 24;
constexpr uint64_t kTilingStride = 512;
constexpr uint64_t kTilingBytes = (kParts + 1) * kTilingStride;
constexpr uint64_t kSystemBytes = 16ULL << 20;

constexpr int64_t kPartColumns[kParts] = {
    kQueryLatent, kKeyValueLatent, kIndexerKey, kRouterGate};

MEGAKERNEL_FRONT_DEVICE constexpr int64_t PartOffset(int64_t part)
{
    int64_t offset = 0;
    for (int64_t earlier = 0; earlier < part; ++earlier) {
        offset += kPartColumns[earlier];
    }
    return offset;
}

// Every core's slice has the same aligned stride.
MEGAKERNEL_FRONT_DEVICE constexpr int64_t SliceStride(int64_t columns,
                                                      int64_t blocks)
{
    const int64_t even = (columns + blocks - 1) / blocks;
    return (even + kColumnGranularity - 1) / kColumnGranularity
        * kColumnGranularity;
}

// A core's contiguous share of `columns`, aligned so the cube sees whole
// fractal blocks. Cores past the end receive no work.
struct Slice {
    int64_t offset, width;
};

MEGAKERNEL_FRONT_DEVICE constexpr Slice SliceFor(int64_t columns, int64_t blocks, int64_t index)
{
    const int64_t stride = SliceStride(columns, blocks);
    const int64_t offset = index * stride;
    if (offset >= columns) {
        return {columns, 0};
    }
    const int64_t remaining = columns - offset;
    return {offset, remaining < stride ? remaining : stride};
}

struct Config {
    int64_t rows, blocks;
    // The weight's row stride. `kColumns` packs the projections adjacently; a
    // larger value spaces them out, which prices the strided read each core
    // makes when it takes a column window of a wide matrix.
    int64_t weight_stride;
};

struct Addresses {
    uint64_t hidden, weight, product, tiling, system;
};

constexpr bool Valid(const Config& config, const Addresses& addresses)
{
    return config.rows >= 1 && config.rows <= kMaxRows
        && config.blocks >= 1 && config.blocks <= kMaxBlocks
        && config.weight_stride >= kColumns
        && config.weight_stride * 2 % kStrideBytes == 0
        && addresses.hidden != 0 && addresses.weight != 0
        && addresses.product != 0 && addresses.tiling != 0
        && addresses.system != 0;
}

}  // namespace megakernel_front

extern "C" int megakernel_front_launch(void* stream, uint64_t sync,
    const megakernel_front::Config* config,
    const megakernel_front::Addresses* addresses);

extern "C" int megakernel_front_build_tiling(int rows, int columns, int width,
                                             int base_m, int base_n, int base_k,
                                             void* buffer);

#endif
