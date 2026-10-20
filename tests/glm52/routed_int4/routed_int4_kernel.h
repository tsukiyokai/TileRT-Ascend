#ifndef TESTS_GLM52_ROUTED_INT4_ROUTED_INT4_KERNEL_H
#define TESTS_GLM52_ROUTED_INT4_ROUTED_INT4_KERNEL_H

#include <cstdint>

#ifndef ROWS_PER_EXPERT
#define ROWS_PER_EXPERT 2
#endif

void routed_int4_launch(uint32_t blocks, void* stream, uint64_t sync_base,
                        void* rows, void* stacked, void* stacked_low,
                        void* gate_up,
                        float* gu_bias, float* gu_scale, void* down,
                        float* down_bias, float* down_scale,
                        float* token_scale, void* partial, void* middle,
                        float* middle_scale, void* wide, void* out,
                        void* gu_tiling, void* down_tiling, void* witness,
                        void* shared_gate_up, float* shared_gu_bias,
                        float* shared_gu_scale, void* shared_down,
                        float* shared_down_bias, float* shared_down_scale,
                        void* shared_out, void* workspace, int64_t experts);

/** Fill a 256-byte buffer with the cube tiling for one INT4 projection. */
extern "C" int routed_int4_build_tiling(int rows, int columns, int inner,
                                        int width, void* buffer);

#endif
