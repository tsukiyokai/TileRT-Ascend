/**
 * Time the TP16 decode reduction as a one-shot push plus ordered local sum.
 *
 * The baseline it has to beat is HCCL: 63.5 microseconds in-graph and 126
 * eager for this message, neither of which varies with the message size. The
 * wire underneath measures 0.31 microseconds within a module.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "shmem.h"
#include "tp_push_kernel.h"
#include "utils.h"

namespace {

constexpr int64_t ALIGN_BYTES = 32;
constexpr int64_t SLOT_INT32 = ALIGN_BYTES / static_cast<int64_t>(sizeof(int32_t));
constexpr uint64_t HEAP_BYTES = 1024UL * 1024UL * 1024UL;

int64_t AlignUp(int64_t value, int64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

}  // namespace

int main(int argc, char* argv[])
{
    if (argc < 8) {
        std::printf("usage: %s <pe_size> <pe_id> <ip_port> <rows> <hidden> <warmup> <rounds> [stage] [blocks]\n", argv[0]);
        return 1;
    }
    const int pe_size = std::atoi(argv[1]);
    const int pe_id = std::atoi(argv[2]);
    const char* ip_port = argv[3];
    const int rows = std::atoi(argv[4]);
    const int hidden = std::atoi(argv[5]);
    const int warmup = std::atoi(argv[6]);
    const int rounds = std::atoi(argv[7]);
    const int stage = (argc > 8) ? std::atoi(argv[8]) : 0;

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(pe_id));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));

    aclshmemx_init_attr_t attributes;
    aclshmemx_uniqueid_t uid{};  // unused by the default config-store bootstrap
    test_set_attr(pe_id, pe_size, HEAP_BYTES, ip_port, uid, &attributes);
    ACL_CHECK(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes));

    const int64_t elements = static_cast<int64_t>(rows) * hidden;
    const int64_t stride = AlignUp(elements * sizeof(float), ALIGN_BYTES) / sizeof(float);
    // Two planes of data and status, alternating by round parity.
    const size_t window_bytes =
        2 * pe_size * stride * sizeof(float) + 2 * pe_size * SLOT_INT32 * sizeof(int32_t);
    void* window = aclshmem_malloc(window_bytes);
    if (window == nullptr) {
        std::printf("pe %d: aclshmem_malloc failed\n", pe_id);
        return 1;
    }
    ACL_CHECK(aclrtMemset(window, window_bytes, 0, window_bytes));

    // Every rank contributes its own index, so the sum is the known constant
    // pe_size*(pe_size-1)/2 in every element and a wrong order cannot pass.
    std::vector<float> host(elements, static_cast<float>(pe_id));
    void* partial = nullptr;
    void* out = nullptr;
    ACL_CHECK(aclrtMalloc(&partial, elements * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&out, elements * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(partial, elements * sizeof(float), host.data(), elements * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE));

    const uint64_t ffts = util_get_ffts_config();
    // The push needs one core per peer; the reduction wants every core it can
    // get, so the launch is sized for the reduction and the push uses the
    // first pe_size cores of it.
    const uint32_t blocks = (argc > 9) ? static_cast<uint32_t>(std::atoi(argv[9])) : 48U;

    // Signal values continue across both launches, so neither needs a reset.
    tp_push_demo(blocks, stream, ffts, (float*)partial, (float*)out, (uint8_t*)window, rows, hidden, warmup, 0, stage);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    aclshmem_barrier_all();

    struct timespec begin;
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &begin);
    tp_push_demo(blocks, stream, ffts, (float*)partial, (float*)out, (uint8_t*)window, rows, hidden, rounds, warmup, stage);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    clock_gettime(CLOCK_MONOTONIC, &end);

    const double total_us =
        (end.tv_sec - begin.tv_sec) * 1e6 + (end.tv_nsec - begin.tv_nsec) / 1e3;
    std::vector<float> result(elements, 0.0F);
    ACL_CHECK(aclrtMemcpy(result.data(), elements * sizeof(float), out, elements * sizeof(float),
                          ACL_MEMCPY_DEVICE_TO_HOST));
    // Only stage 0 produces a sum; the push-free stages leave `out` untouched,
    // so checking them against the sum would always report every element wrong.
    const float expected = static_cast<float>(pe_size * (pe_size - 1) / 2);
    int64_t wrong = 0;
    if (stage == 0) {
        for (int64_t i = 0; i < elements; ++i) {
            if (result[i] != expected) {
                ++wrong;
            }
        }
    }
    if (pe_id == 0) {
        std::printf("pe_size %d rows %d hidden %d bytes %ld rounds %d blocks %u\n", pe_size, rows,
                    hidden, (long)(elements * sizeof(float)), rounds, blocks);
        std::printf("RESULT stage %d per_reduction_us %.3f wrong %ld of %ld expected %.1f got %.1f\n",
                    stage, total_us / rounds, (long)wrong, (long)elements, expected, result[0]);
    }

    aclshmem_free(window);
    aclshmem_finalize();
    ACL_CHECK(aclrtFree(partial));
    ACL_CHECK(aclrtFree(out));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(pe_id));
    ACL_CHECK(aclFinalize());
    return (stage == 0 && wrong != 0) ? 1 : 0;
}
