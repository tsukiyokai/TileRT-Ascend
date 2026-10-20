/**
 * Does one MIX kernel beat a matmul kernel plus a vector kernel?
 *
 * Both arms run in this process on the same inputs, alternating, at the same
 * block count, so the comparison is of the cube-to-vector boundary and not of
 * a parallel decomposition. Correctness is against a host reference that
 * computes the INT32 product exactly and rounds to BF16 once, where the
 * device rounds.
 */
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

#include "acl/acl.h"
#include "mix_expert_kernel.h"

// Not in the public headers; the SHMEM tree reaches it the same way.
extern "C" int32_t rtGetC2cCtrlAddr(uint64_t* addr, uint32_t* len);

namespace {

constexpr int64_t HIDDEN = 6144;
constexpr int64_t HALF = 128;
constexpr int64_t GATE_UP = 2 * HALF;
constexpr size_t TILING_BYTES = 256;
constexpr uint32_t BLOCK_DIM = 1;

void Check(aclError status, const char* what)
{
    if (status != ACL_SUCCESS) {
        std::printf("%s failed: %d\n", what, status);
        std::exit(1);
    }
}

double Elapsed(const timespec& begin, const timespec& end)
{
    return (end.tv_sec - begin.tv_sec) * 1e6
           + (end.tv_nsec - begin.tv_nsec) / 1e3;
}

uint16_t ToBf16(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t low = bits & 0xFFFF;
    const uint32_t carry = (low > 0x8000)
                           || (low == 0x8000 && ((bits >> 16) & 1));
    return static_cast<uint16_t>((bits >> 16) + carry);
}

}  // namespace

int main(int argc, char** argv)
{
    const int64_t rows = (argc > 1) ? std::atoll(argv[1]) : 8;
    const int32_t device = (argc > 2) ? std::atoi(argv[2]) : 0;
    const int repeats = (argc > 3) ? std::atoi(argv[3]) : 100;
    const int rounds = (argc > 4) ? std::atoi(argv[4]) : 5;
    // 0 both arms, 1 the staged arm alone, 2 the fused arm alone. A hang
    // tells you nothing unless you can run one side of it.
    const int arm = (argc > 5) ? std::atoi(argv[5]) : 0;
    const int stages = (argc > 6) ? std::atoi(argv[6]) : 7;

    std::vector<uint8_t> tiling_host(TILING_BYTES, 0);
    const int tiling_status = mix_build_tiling(static_cast<int>(rows),
                                               tiling_host.data());
    if (tiling_status != 0) {
        std::printf("mix_build_tiling returned %d\n", tiling_status);
        return 1;
    }

    Check(aclInit(nullptr), "aclInit");
    Check(aclrtSetDevice(device), "aclrtSetDevice");
    aclrtStream stream = nullptr;
    Check(aclrtCreateStream(&stream), "aclrtCreateStream");

    // The control address exists once a device is bound, not before.
    uint64_t sync_base = 0;
    uint32_t sync_length = 0;
    const int32_t sync_status = rtGetC2cCtrlAddr(&sync_base, &sync_length);
    if (sync_status != 0) {
        std::printf("rtGetC2cCtrlAddr returned %d\n", sync_status);
        return 1;
    }

    std::vector<int8_t> x_host(rows * HIDDEN);
    std::vector<int8_t> weight_host(GATE_UP * HIDDEN);
    std::vector<float> scale_host(GATE_UP);
    std::vector<float> token_host(rows);
    for (int64_t index = 0; index < rows * HIDDEN; ++index) {
        x_host[index] = static_cast<int8_t>((index % 37) - 18);
    }
    for (int64_t index = 0; index < GATE_UP * HIDDEN; ++index) {
        weight_host[index] = static_cast<int8_t>((index % 23) - 11);
    }
    for (int64_t index = 0; index < GATE_UP; ++index) {
        scale_host[index] = 0.00031F + 0.0000007F * static_cast<float>(index);
    }
    for (int64_t index = 0; index < rows; ++index) {
        token_host[index] = 0.021F + 0.003F * static_cast<float>(index);
    }

    void* x = nullptr;
    void* weight = nullptr;
    void* partial = nullptr;
    void* scale = nullptr;
    void* token = nullptr;
    void* fused_out = nullptr;
    void* staged_out = nullptr;
    void* tiling = nullptr;
    Check(aclrtMalloc(&x, x_host.size(), ACL_MEM_MALLOC_HUGE_FIRST), "x");
    Check(aclrtMalloc(&weight, weight_host.size(),
                      ACL_MEM_MALLOC_HUGE_FIRST), "weight");
    Check(aclrtMalloc(&partial, rows * GATE_UP * sizeof(int32_t),
                      ACL_MEM_MALLOC_HUGE_FIRST), "partial");
    Check(aclrtMalloc(&scale, GATE_UP * sizeof(float),
                      ACL_MEM_MALLOC_HUGE_FIRST), "scale");
    Check(aclrtMalloc(&token, rows * sizeof(float),
                      ACL_MEM_MALLOC_HUGE_FIRST), "token");
    Check(aclrtMalloc(&fused_out, rows * HALF * sizeof(uint16_t),
                      ACL_MEM_MALLOC_HUGE_FIRST), "fused out");
    Check(aclrtMalloc(&staged_out, rows * HALF * sizeof(uint16_t),
                      ACL_MEM_MALLOC_HUGE_FIRST), "staged out");
    Check(aclrtMalloc(&tiling, TILING_BYTES, ACL_MEM_MALLOC_HUGE_FIRST),
          "tiling");

    Check(aclrtMemcpy(x, x_host.size(), x_host.data(), x_host.size(),
                      ACL_MEMCPY_HOST_TO_DEVICE), "upload x");
    Check(aclrtMemcpy(weight, weight_host.size(), weight_host.data(),
                      weight_host.size(), ACL_MEMCPY_HOST_TO_DEVICE),
          "upload weight");
    Check(aclrtMemcpy(scale, GATE_UP * sizeof(float), scale_host.data(),
                      GATE_UP * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE),
          "upload scale");
    Check(aclrtMemcpy(token, rows * sizeof(float), token_host.data(),
                      rows * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE),
          "upload token");
    Check(aclrtMemcpy(tiling, TILING_BYTES, tiling_host.data(), TILING_BYTES,
                      ACL_MEMCPY_HOST_TO_DEVICE), "upload tiling");

    if (arm != 2) {
        std::printf("staged launching\n");
        std::fflush(stdout);
        mix_expert_staged_launch(BLOCK_DIM, stream, sync_base, x, weight,
                                 partial, (float*)scale, (float*)token,
                                 staged_out, tiling, rows);
        Check(aclrtSynchronizeStream(stream), "staged once");
        std::printf("staged returned\n");
        std::fflush(stdout);
    }
    if (arm != 1) {
        std::printf("fused launching\n");
        std::fflush(stdout);
        mix_expert_fused_launch(BLOCK_DIM, stream, sync_base, x, weight,
                                partial, (float*)scale, (float*)token,
                                fused_out, tiling, rows, stages);
        Check(aclrtSynchronizeStream(stream), "fused once");
        std::printf("fused returned\n");
        std::fflush(stdout);
    }
    if (arm != 0) {
        Check(aclrtDestroyStream(stream), "destroy");
        Check(aclrtResetDevice(device), "reset");
        Check(aclFinalize(), "finalize");
        return 0;
    }

    std::vector<uint16_t> fused_host(rows * HALF);
    std::vector<uint16_t> staged_host(rows * HALF);
    const size_t out_bytes = rows * HALF * sizeof(uint16_t);
    Check(aclrtMemcpy(fused_host.data(), out_bytes, fused_out, out_bytes,
                      ACL_MEMCPY_DEVICE_TO_HOST), "download fused");
    Check(aclrtMemcpy(staged_host.data(), out_bytes, staged_out, out_bytes,
                      ACL_MEMCPY_DEVICE_TO_HOST), "download staged");

    int64_t wrong_reference = 0;
    int64_t wrong_arms = 0;
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t column = 0; column < HALF; ++column) {
            int32_t gate_product = 0;
            int32_t up_product = 0;
            for (int64_t inner = 0; inner < HIDDEN; ++inner) {
                const int32_t activation = x_host[row * HIDDEN + inner];
                gate_product += activation
                                * weight_host[column * HIDDEN + inner];
                up_product += activation
                              * weight_host[(HALF + column) * HIDDEN + inner];
            }
            const float gate = static_cast<float>(gate_product)
                               * scale_host[column] * token_host[row];
            const float up = static_cast<float>(up_product)
                             * scale_host[HALF + column] * token_host[row];
            const float want = gate / (1.0F + std::exp(-gate)) * up;
            const uint16_t rounded = ToBf16(want);
            const int64_t index = row * HALF + column;
            // SwiGLU is a transcendental, so one unit in the last place of
            // BF16 is agreement; the INT32 product before it is exact.
            const int32_t slack = static_cast<int32_t>(fused_host[index])
                                  - static_cast<int32_t>(rounded);
            wrong_reference += (slack > 1 || slack < -1);
            wrong_arms += (fused_host[index] != staged_host[index]);
        }
    }
    std::printf("rows %lld  blocks %u  tiling %d\n", (long long)rows,
                BLOCK_DIM, tiling_status);
    std::printf("fused against host reference, 1 ulp allowed: %lld of %lld\n",
                (long long)wrong_reference, (long long)(rows * HALF));
    std::printf("fused against staged arm, exact:             %lld of %lld\n",
                (long long)wrong_arms, (long long)(rows * HALF));

    for (int round = 0; round < rounds; ++round) {
        timespec begin;
        timespec end;
        clock_gettime(CLOCK_MONOTONIC, &begin);
        for (int step = 0; step < repeats; ++step) {
            mix_expert_fused_launch(BLOCK_DIM, stream, sync_base, x, weight,
                                    partial, (float*)scale, (float*)token,
                                    fused_out, tiling, rows);
        }
        Check(aclrtSynchronizeStream(stream), "fused timing");
        clock_gettime(CLOCK_MONOTONIC, &end);
        const double fused = Elapsed(begin, end) / repeats;

        clock_gettime(CLOCK_MONOTONIC, &begin);
        for (int step = 0; step < repeats; ++step) {
            mix_expert_staged_launch(BLOCK_DIM, stream, sync_base, x, weight,
                                     partial, (float*)scale, (float*)token,
                                     staged_out, tiling, rows);
        }
        Check(aclrtSynchronizeStream(stream), "staged timing");
        clock_gettime(CLOCK_MONOTONIC, &end);
        const double staged = Elapsed(begin, end) / repeats;

        std::printf("round %d  fused %8.3f us  staged %8.3f us  saved "
                    "%7.3f  %.2fx\n", round, fused, staged, staged - fused,
                    staged / fused);
    }

    Check(aclrtDestroyStream(stream), "destroy");
    Check(aclrtResetDevice(device), "reset");
    Check(aclFinalize(), "finalize");
    return (wrong_reference == 0 && wrong_arms == 0) ? 0 : 1;
}
