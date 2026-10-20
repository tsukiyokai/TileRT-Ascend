#ifndef TESTS_GLM52_MIXBRICK_MIX_EXPERT_KERNEL_H
#define TESTS_GLM52_MIXBRICK_MIX_EXPERT_KERNEL_H

#include <cstdint>

void mix_expert_fused_launch(uint32_t blocks, void* stream, uint64_t sync_base,
                             void* x, void* gate_up, void* partial,
                             float* gu_scale, float* token_scale, void* out,
                             void* tiling, int64_t rows, int64_t stages);

void mix_expert_staged_launch(uint32_t blocks, void* stream,
                              uint64_t sync_base, void* x, void* gate_up,
                              void* partial, float* gu_scale,
                              float* token_scale, void* out, void* tiling,
                              int64_t rows);

/** Fill a 200-byte buffer with the cube tiling for this row count. */
extern "C" int mix_build_tiling(int rows, int channels, void* buffer);

#endif
