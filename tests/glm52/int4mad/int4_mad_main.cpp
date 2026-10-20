/**
 * Both operands are INT4 ones, so every element of the product should be K, and
 * that is true in any fractal order. The second filling makes the nibbles of
 * a byte differ: 0x21 is a one and a two, so the sum over K is 96 if both
 * halves are read, 64 or 128 if only one is.
 */
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "acl/acl.h"
#include "int4_mad_kernel.h"

namespace {

constexpr int64_t ROWS = 16;
constexpr int64_t INNER = 64;
constexpr int64_t COLUMNS = 16;

void Check(aclError status, const char* what)
{
    if (status != ACL_SUCCESS) {
        std::printf("%s failed: %d\n", what, status);
        std::exit(2);
    }
}

}  // namespace

int main(int argc, char** argv)
{
    const uint8_t filling = (argc > 1) ? (uint8_t)std::strtol(argv[1], nullptr, 16)
                                       : 0x11;
    const int32_t device = (argc > 2) ? std::atoi(argv[2]) : 0;

    Check(aclInit(nullptr), "aclInit");
    Check(aclrtSetDevice(device), "aclrtSetDevice");
    aclrtStream stream = nullptr;
    Check(aclrtCreateStream(&stream), "aclrtCreateStream");

    void* left = nullptr;
    void* right = nullptr;
    void* out = nullptr;
    Check(aclrtMalloc(&left, ROWS * INNER / 2, ACL_MEM_MALLOC_HUGE_FIRST),
          "left");
    Check(aclrtMalloc(&right, COLUMNS * INNER / 2, ACL_MEM_MALLOC_HUGE_FIRST),
          "right");
    Check(aclrtMalloc(&out, ROWS * COLUMNS * sizeof(int32_t),
                      ACL_MEM_MALLOC_HUGE_FIRST), "out");

    const std::vector<uint8_t> left_host(ROWS * INNER / 2, 0x11);
    const std::vector<uint8_t> right_host(COLUMNS * INNER / 2, filling);
    const std::vector<int32_t> sentinel(ROWS * COLUMNS, 0x7777);
    Check(aclrtMemcpy(left, left_host.size(), left_host.data(),
                      left_host.size(), ACL_MEMCPY_HOST_TO_DEVICE), "up left");
    Check(aclrtMemcpy(right, right_host.size(), right_host.data(),
                      right_host.size(), ACL_MEMCPY_HOST_TO_DEVICE),
          "up right");
    Check(aclrtMemcpy(out, sentinel.size() * sizeof(int32_t), sentinel.data(),
                      sentinel.size() * sizeof(int32_t),
                      ACL_MEMCPY_HOST_TO_DEVICE), "up out");

    int4_mad_launch(stream, left, right, out);
    Check(aclrtSynchronizeStream(stream), "synchronize");

    std::vector<int32_t> got(ROWS * COLUMNS);
    Check(aclrtMemcpy(got.data(), got.size() * sizeof(int32_t), out,
                      got.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST),
          "download");

    int32_t lowest = got[0];
    int32_t highest = got[0];
    int64_t untouched = 0;
    for (const int32_t one : got) {
        lowest = (one < lowest) ? one : lowest;
        highest = (one > highest) ? one : highest;
        untouched += (one == 0x7777);
    }
    const int64_t expected = (filling == 0x11) ? INNER : 3 * INNER / 2;
    std::printf("right filled 0x%02x  expect %lld  got %d to %d  "
                "(%lld untouched)  first row:", filling, (long long)expected,
                lowest, highest, (long long)untouched);
    for (int64_t column = 0; column < 8; ++column) {
        std::printf(" %d", got[column]);
    }
    std::printf("\n");

    Check(aclrtDestroyStream(stream), "destroy");
    Check(aclrtResetDevice(device), "reset");
    Check(aclFinalize(), "finalize");
    return (lowest == expected && highest == expected) ? 0 : 1;
}
