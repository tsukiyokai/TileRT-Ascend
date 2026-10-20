/**
 * The same chain with INT4 weights, and the same question.
 *
 * The reference stays a plain INT8-against-INT4 matmul, because splitting the
 * activation into two INT4 halves is exact: 16 * high + low is the INT8 back
 * again. So any deviation here is a fault in the split, the stacking or the
 * recombination rather than something the reference has to model.
 *
 * Activations are generated within 119 of zero, which is what two signed INT4
 * can carry as 16 * high + low, and the kernel's own quantisation divides by
 * 119 for the same reason.
 */
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

#include "acl/acl.h"
#include "routed_int4_kernel.h"

extern "C" int32_t rtGetC2cCtrlAddr(uint64_t* addr, uint32_t* len);

namespace {

constexpr int64_t HIDDEN = 6144;
constexpr int64_t HALF = 128;
constexpr int64_t GATE_UP = 2 * HALF;
#ifndef ROWS_PER_EXPERT
#define ROWS_PER_EXPERT 2
#endif
constexpr int64_t ROWS = ROWS_PER_EXPERT;
constexpr int64_t STACKED = 2 * ROWS;
// The cube's row tile, which the library reads and writes in full.
constexpr int64_t M_TILE = 16;
constexpr float QUANTUM_SPAN = 127.0F;
constexpr int64_t BLOCK_FLOATS = 8;
constexpr int64_t SCALE_STRIDE = ((ROWS + BLOCK_FLOATS - 1) / BLOCK_FLOATS)
                                 * BLOCK_FLOATS;
constexpr size_t TILING_BYTES = 256;
// One tile of INT4 a expert, which is how wide the kernel's stages work.
constexpr int64_t TILE_BYTES = 2048 / 2;  // one tile of INT4

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

/** Round to IEEE half, which the device's INT8 cast goes through. */
float ToHalf(float value)
{
    const float magnitude = std::fabs(value);
    if (magnitude < 6.103515625e-05F) {
        return value;
    }
    int exponent = 0;
    std::frexp(magnitude, &exponent);
    const float quantum = std::ldexp(1.0F, exponent - 11);
    const float rounded = std::nearbyintf(magnitude / quantum) * quantum;
    return (value < 0.0F) ? -rounded : rounded;
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

void* Hold(size_t bytes, const char* what)
{
    void* block = nullptr;
    Check(aclrtMalloc(&block, bytes, ACL_MEM_MALLOC_HUGE_FIRST), what);
    return block;
}

}  // namespace

int main(int argc, char** argv)
{
    const int64_t experts = (argc > 1) ? std::atoll(argv[1]) : 8;
    const int32_t device = (argc > 2) ? std::atoi(argv[2]) : 0;
    const uint32_t blocks = (argc > 3) ? std::atoi(argv[3]) : 4;
    const int repeats = (argc > 4) ? std::atoi(argv[4]) : 20;
    const int rounds = (argc > 5) ? std::atoi(argv[5]) : 3;

    std::vector<uint8_t> gu_tiling_host(TILING_BYTES, 0);
    std::vector<uint8_t> down_tiling_host(TILING_BYTES, 0);
    const int gu_status = routed_int4_build_tiling(STACKED, GATE_UP, HIDDEN, GATE_UP,
                                              gu_tiling_host.data());
    const int down_status = routed_int4_build_tiling(STACKED, HIDDEN, HALF, HIDDEN,
                                                down_tiling_host.data());
    if (gu_status != 0 || down_status != 0) {
        std::printf("tiling returned %d and %d\n", gu_status, down_status);
        return 1;
    }

    Check(aclInit(nullptr), "aclInit");
    Check(aclrtSetDevice(device), "aclrtSetDevice");
    aclrtStream stream = nullptr;
    Check(aclrtCreateStream(&stream), "aclrtCreateStream");
    uint64_t sync_base = 0;
    uint32_t sync_length = 0;
    const int32_t sync_status = rtGetC2cCtrlAddr(&sync_base, &sync_length);
    if (sync_status != 0) {
        std::printf("rtGetC2cCtrlAddr returned %d\n", sync_status);
        return 1;
    }

    std::vector<int8_t> rows_host(experts * ROWS * HIDDEN);
    std::vector<int8_t> gu_host(experts * GATE_UP * HIDDEN);  // one INT4 a slot
    std::vector<int8_t> down_host(experts * HIDDEN * HALF);
    std::vector<float> gu_scale_host(experts * GATE_UP);
    std::vector<float> down_scale_host(experts * HIDDEN);
    std::vector<float> token_host(experts * SCALE_STRIDE, 0.0F);
    // Pseudo-random rather than an arithmetic progression. A progression's
    // low nibbles are a sawtooth of period sixteen, which matches itself at
    // every aligned phase -- a search for where a half actually landed then
    // answers almost anywhere, and a wrong answer can hide behind a right
    // one. The whole INT8 range is exact under this split, so the generator
    // uses all of it; weights are INT4, so within 8 of zero.
    uint32_t seed = 0x5eed1234U;
    const auto next = [&seed]() {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        return seed;
    };
    for (size_t index = 0; index < rows_host.size(); ++index) {
        rows_host[index] = static_cast<int8_t>((int32_t)(next() % 255) - 127);
    }
    for (size_t index = 0; index < gu_host.size(); ++index) {
        gu_host[index] = static_cast<int8_t>((int32_t)(next() % 15) - 7);
    }
    for (size_t index = 0; index < down_host.size(); ++index) {
        down_host[index] = static_cast<int8_t>((int32_t)(next() % 15) - 7);
    }
    // K-major copies for the device, since an INT4 B operand cannot be
    // transposed on the way in: `load_cbuf_to_cb_s4` hardcodes transpose off
    // and drops the flag without a word.
    const auto transpose = [](const std::vector<int8_t>& source,
                              int64_t groups, int64_t rows, int64_t columns) {
        std::vector<int8_t> turned(source.size());
        for (int64_t group = 0; group < groups; ++group) {
            for (int64_t row = 0; row < rows; ++row) {
                for (int64_t column = 0; column < columns; ++column) {
                    turned[(group * columns + column) * rows + row] =
                        source[(group * rows + row) * columns + column];
                }
            }
        }
        return turned;
    };
    const std::vector<int8_t> gu_kmajor =
        transpose(gu_host, experts, GATE_UP, HIDDEN);
    const std::vector<int8_t> down_kmajor =
        transpose(down_host, experts, HIDDEN, HALF);
    for (size_t index = 0; index < gu_scale_host.size(); ++index) {
        gu_scale_host[index] = 0.00031F + 0.0000007F * (index % 256);
    }
    for (size_t index = 0; index < down_scale_host.size(); ++index) {
        down_scale_host[index] = 0.00047F + 0.0000003F * (index % 6144);
    }
    // Every low INT4 half carries an eight that has to come back, and the
    // sum it comes back as depends only on the weight -- so it is computed
    // here once, as a per-column constant, the way a loader would.
    std::vector<float> gu_bias_host(experts * GATE_UP, 0.0F);
    std::vector<float> down_bias_host(experts * HIDDEN, 0.0F);
    for (int64_t expert = 0; expert < experts; ++expert) {
        for (int64_t column = 0; column < GATE_UP; ++column) {
            int32_t total = 0;
            for (int64_t inner = 0; inner < HIDDEN; ++inner) {
                total += gu_host[(expert * GATE_UP + column) * HIDDEN
                                 + inner];
            }
            gu_bias_host[expert * GATE_UP + column] = 8.0F * (float)total;
        }
        for (int64_t column = 0; column < HIDDEN; ++column) {
            int32_t total = 0;
            for (int64_t inner = 0; inner < HALF; ++inner) {
                total += down_host[(expert * HIDDEN + column) * HALF + inner];
            }
            down_bias_host[expert * HIDDEN + column] = 8.0F * (float)total;
        }
    }

    for (int64_t expert = 0; expert < experts; ++expert) {
        for (int64_t row = 0; row < ROWS; ++row) {
            token_host[expert * SCALE_STRIDE + row] =
                0.021F + 0.003F * static_cast<float>(row)
                + 0.0005F * static_cast<float>(expert);
        }
    }

    void* rows = Hold(rows_host.size(), "rows");
    void* gate_up = Hold(gu_host.size() / 2, "gate up");
    void* down = Hold(down_host.size() / 2, "down");
    void* gu_bias = Hold(gu_bias_host.size() * sizeof(float), "gu bias");
    void* down_bias = Hold(down_bias_host.size() * sizeof(float), "d bias");
    void* gu_scale = Hold(gu_scale_host.size() * sizeof(float), "gu scale");
    void* down_scale = Hold(down_scale_host.size() * sizeof(float), "d scale");
    void* token = Hold(token_host.size() * sizeof(float), "token");
    void* stacked = Hold(experts * M_TILE * HIDDEN / 2, "stacked");
    // The low half's base, offset on the host rather than on the device.
    void* stacked_low = (void*)((char*)stacked + ROWS * HIDDEN / 2);
    void* partial = Hold(experts * M_TILE * GATE_UP * sizeof(int32_t),
                         "part");
    void* middle = Hold(experts * M_TILE * HALF / 2, "middle");
    void* middle_scale = Hold(experts * SCALE_STRIDE * sizeof(float), "mid s");
    void* wide = Hold(experts * M_TILE * HIDDEN * sizeof(int32_t), "wide");
    void* out = Hold(experts * ROWS * HIDDEN * sizeof(uint16_t), "out");
    void* gu_tiling = Hold(TILING_BYTES, "gu tiling");
    void* down_tiling = Hold(TILING_BYTES, "down tiling");
    void* witness = Hold(experts * TILE_BYTES, "witness");
    // Scratch for the Matmul library, which takes it from the system
    // workspace. Sixteen mebibytes is far more than its tiling asks for.
    void* workspace = Hold(16u << 20, "workspace");

    const auto upload = [&](void* where, const void* what, size_t bytes,
                            const char* name) {
        Check(aclrtMemcpy(where, bytes, what, bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE), name);
    };
    upload(rows, rows_host.data(), rows_host.size(), "rows");
    const auto pack = [](const std::vector<int8_t>& wide_form) {
        std::vector<uint8_t> packed(wide_form.size() / 2, 0);
        for (size_t index = 0; index < wide_form.size(); ++index) {
            const uint8_t nibble = (uint8_t)(wide_form[index] & 0x0F);
            packed[index / 2] |= (index % 2 == 0) ? nibble : (nibble << 4);
        }
        return packed;
    };
    const std::vector<uint8_t> gu_packed = pack(gu_kmajor);
    const std::vector<uint8_t> down_packed = pack(down_kmajor);
    upload(gate_up, gu_packed.data(), gu_packed.size(), "gate up");
    upload(down, down_packed.data(), down_packed.size(), "down");
    upload(gu_bias, gu_bias_host.data(),
           gu_bias_host.size() * sizeof(float), "gu bias");
    upload(down_bias, down_bias_host.data(),
           down_bias_host.size() * sizeof(float), "down bias");
    upload(gu_scale, gu_scale_host.data(),
           gu_scale_host.size() * sizeof(float), "gu scale");
    upload(down_scale, down_scale_host.data(),
           down_scale_host.size() * sizeof(float), "down scale");
    upload(token, token_host.data(), token_host.size() * sizeof(float),
           "token");
    upload(gu_tiling, gu_tiling_host.data(), TILING_BYTES, "gu tiling");
    upload(down_tiling, down_tiling_host.data(), TILING_BYTES, "down tiling");

    // A sentinel separates a wrong value from an unwritten one.
    const uint16_t SENTINEL = 0x7777;
    const std::vector<uint16_t> sentinel(experts * ROWS * HIDDEN, SENTINEL);
    upload(out, sentinel.data(), sentinel.size() * sizeof(uint16_t), "out");

    routed_int4_launch(blocks, stream, sync_base, rows, stacked, stacked_low,
                       gate_up,
                       (float*)gu_bias, (float*)gu_scale, down,
                       (float*)down_bias, (float*)down_scale, (float*)token,
                       partial, middle, (float*)middle_scale, wide, out,
                       gu_tiling, down_tiling, witness, workspace, experts);
    Check(aclrtSynchronizeStream(stream), "once");

    std::vector<uint16_t> out_host(experts * ROWS * HIDDEN);
    Check(aclrtMemcpy(out_host.data(), out_host.size() * sizeof(uint16_t),
                      out, out_host.size() * sizeof(uint16_t),
                      ACL_MEMCPY_DEVICE_TO_HOST), "download out");

    std::vector<int32_t> partial_host(experts * M_TILE * GATE_UP);
    Check(aclrtMemcpy(partial_host.data(),
                      partial_host.size() * sizeof(int32_t), partial,
                      partial_host.size() * sizeof(int32_t),
                      ACL_MEMCPY_DEVICE_TO_HOST), "download partial");
    std::vector<uint8_t> stacked_host(experts * M_TILE * HIDDEN / 2);
    Check(aclrtMemcpy(stacked_host.data(), stacked_host.size(), stacked,
                      stacked_host.size(), ACL_MEMCPY_DEVICE_TO_HOST),
          "download stacked");
    {
        const auto nibble = [&](size_t index) {
            const uint8_t byte = stacked_host[index / 2];
            const uint8_t four = (index % 2 == 0) ? (byte & 0x0F)
                                                  : (byte >> 4);
            return (int)((four >= 8) ? (int)four - 16 : (int)four);
        };
        std::printf("row 0 of expert 0, first 8 of K:\n  int8  ");
        for (int64_t at = 0; at < 8; ++at) {
            std::printf(" %4d", rows_host[at]);
        }
        std::printf("\n  high ");
        for (int64_t at = 0; at < 8; ++at) {
            std::printf(" %4d", nibble(at));
        }
        std::printf("\n  low  ");
        for (int64_t at = 0; at < 8; ++at) {
            std::printf(" %4d", nibble(ROWS * HIDDEN + at));
        }
        std::printf("\n  16*high+low");
        for (int64_t at = 0; at < 8; ++at) {
            std::printf(" %4d", 16 * nibble(at)
                                + nibble(ROWS * HIDDEN + at) + 8);
        }
        // Where did the low halves actually land? Searching for them beats
        // guessing why they are not where they were addressed.
        {
            std::vector<int> want_low(32);
            for (int64_t at = 0; at < 32; ++at) {
                const int32_t x = rows_host[at];
                const int32_t high = (int32_t)std::floor(x / 16.0);
                want_low[at] = x - 16 * high - 8;
            }
            int64_t found_at = -1;
            const int64_t slots = (int64_t)stacked_host.size() * 2;
            for (int64_t start = 0; start + 32 < slots && found_at < 0;
                 start += 2) {
                bool all = true;
                for (int64_t at = 0; at < 32 && all; ++at) {
                    all = (nibble(start + at) == want_low[at]);
                }
                if (all) {
                    found_at = start;
                }
            }
            std::vector<int> want_high(32);
            for (int64_t at = 0; at < 32; ++at) {
                want_high[at] = (int32_t)std::floor(rows_host[at] / 16.0);
            }
            int64_t high_at = -1;
            for (int64_t start = 0; start + 32 < slots && high_at < 0;
                 start += 2) {
                bool all = true;
                for (int64_t at = 0; at < 32 && all; ++at) {
                    all = (nibble(start + at) == want_high[at]);
                }
                if (all) {
                    high_at = start;
                }
            }
            std::printf("\n  high halves addressed at 0, found at %lld"
                        "\n  low halves addressed at nibble %lld, found at "
                        "%lld", (long long)high_at,
                        (long long)(ROWS * HIDDEN), (long long)found_at);
        }
        std::vector<uint8_t> witness_host(experts * TILE_BYTES);
        Check(aclrtMemcpy(witness_host.data(), witness_host.size(), witness,
                          witness_host.size(), ACL_MEMCPY_DEVICE_TO_HOST),
              "down witness");
        // Identical bytes at 6144 and 12288 say the region repeats, so look
        // at every boundary the writes could have used.
        for (int64_t where : {0L, 1024L, 3072L, 6144L, 9216L, 12288L}) {
            std::printf("\n  raw bytes at %5ld:", where);
            for (int64_t at = 0; at < 6; ++at) {
                std::printf(" %02x", stacked_host[where + at]);
            }
        }
        std::printf("\n  low halves through a dedicated buffer at zero:");
        for (int64_t at = 0; at < 8; ++at) {
            const uint8_t byte = witness_host[at / 2];
            const uint8_t four = (at % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
            std::printf(" %4d", (int)((four >= 8) ? (int)four - 16
                                                  : (int)four));
        }
        // The first projection is exact, so walk the next stage: the
        // quantised intermediate, its per-row scale, and the down product.
        std::vector<uint8_t> middle_host(experts * M_TILE * HALF / 2);
        Check(aclrtMemcpy(middle_host.data(), middle_host.size(), middle,
                          middle_host.size(), ACL_MEMCPY_DEVICE_TO_HOST),
              "down middle");
        std::vector<float> middle_scale_host(experts * SCALE_STRIDE);
        Check(aclrtMemcpy(middle_scale_host.data(),
                          middle_scale_host.size() * sizeof(float),
                          middle_scale, middle_scale_host.size()
                          * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST),
              "down middle scale");
        {
            const auto four = [&](int64_t index) {
                const uint8_t byte = middle_host[index / 2];
                const uint8_t nibble = (index % 2 == 0) ? (byte & 0x0F)
                                                        : (byte >> 4);
                return (int)((nibble >= 8) ? (int)nibble - 16 : (int)nibble);
            };
            std::printf("\n  intermediate 16*high+low, first 8:");
            for (int64_t at = 0; at < 8; ++at) {
                std::printf(" %4d", 16 * four(at)
                                    + four(ROWS * HALF + at) + 8);
            }
            std::printf("\n  its row scale: %g %g", middle_scale_host[0],
                        middle_scale_host[1]);
        }
        // Row zero's scale matches the reference exactly and row one's does
        // not, so check the projection a row at a time.
        for (int64_t row = 0; row < ROWS; ++row) {
            std::printf("\n  row %lld product: want", (long long)row);
            for (int64_t column = 0; column < 3; ++column) {
                int32_t want = 0;
                for (int64_t inner = 0; inner < HIDDEN; ++inner) {
                    want += rows_host[row * HIDDEN + inner]
                            * gu_host[column * HIDDEN + inner];
                }
                std::printf(" %d", want);
            }
            std::printf("  got");
            for (int64_t column = 0; column < 3; ++column) {
                const int32_t high = partial_host[row * GATE_UP + column];
                const int32_t low =
                    partial_host[(ROWS + row) * GATE_UP + column];
                std::printf(" %d", 16 * high + low
                                   + (int32_t)gu_bias_host[column]);
            }
        }
        std::printf("\n  product, first 4 columns: want");
        for (int64_t column = 0; column < 4; ++column) {
            int32_t want = 0;
            for (int64_t inner = 0; inner < HIDDEN; ++inner) {
                want += rows_host[inner] * gu_host[column * HIDDEN + inner];
            }
            std::printf(" %d", want);
        }
        std::printf("  got");
        for (int64_t column = 0; column < 4; ++column) {
            const int32_t high = partial_host[column];
            const int32_t low = partial_host[ROWS * GATE_UP + column];
            std::printf(" %d", 16 * high + low
                              + (int32_t)gu_bias_host[column]);
        }
        std::printf("\n");
    }

    int64_t wrong = 0;
    int64_t untouched = 0;
    std::vector<int64_t> deviating_row(ROWS, 0);
    // Narrow splits the 6144 columns between the two vector subblocks, so
    // count the halves separately to say which subblock is wrong.
    int64_t deviating_low_half = 0;
    int64_t deviating_high_half = 0;
    std::vector<float> activation(HALF);
    std::vector<int8_t> code(HALF);
    for (int64_t expert = 0; expert < experts; ++expert) {
        for (int64_t row = 0; row < ROWS; ++row) {
            float largest = 0.0F;
            for (int64_t column = 0; column < HALF; ++column) {
                int32_t gate_product = 0;
                int32_t up_product = 0;
                for (int64_t inner = 0; inner < HIDDEN; ++inner) {
                    const int32_t one =
                        rows_host[(expert * ROWS + row) * HIDDEN + inner];
                    gate_product += one * gu_host[
                        (expert * GATE_UP + column) * HIDDEN + inner];
                    up_product += one * gu_host[
                        (expert * GATE_UP + HALF + column) * HIDDEN + inner];
                }
                const float of_row =
                    token_host[expert * SCALE_STRIDE + row];
                const float gate = static_cast<float>(gate_product)
                                   * gu_scale_host[expert * GATE_UP + column]
                                   * of_row;
                const float up = static_cast<float>(up_product)
                                 * gu_scale_host[expert * GATE_UP + HALF
                                                 + column] * of_row;
                const float value = gate / (1.0F + std::exp(-gate)) * up;
                activation[column] = value;
                const float magnitude = std::fabs(value);
                largest = (magnitude > largest) ? magnitude : largest;
            }
            const float quantum = largest / QUANTUM_SPAN;
            for (int64_t column = 0; column < HALF; ++column) {
                code[column] = static_cast<int8_t>(
                    std::nearbyintf(activation[column] / quantum));
            }
            if (expert == 0 && row < 2) {
                std::printf("  reference row %lld: quantum %g  code",
                            (long long)row, quantum);
                for (int64_t at = 0; at < 8; ++at) {
                    std::printf(" %4d", (int)code[at]);
                }
                std::printf("\n");
            }
            for (int64_t column = 0; column < HIDDEN; ++column) {
                int32_t product = 0;
                for (int64_t inner = 0; inner < HALF; ++inner) {
                    product += code[inner] * down_host[
                        (expert * HIDDEN + column) * HALF + inner];
                }
                const float want = static_cast<float>(product)
                                   * down_scale_host[expert * HIDDEN + column]
                                   * quantum;
                const uint16_t got =
                    out_host[(expert * ROWS + row) * HIDDEN + column];
                untouched += (got == SENTINEL);
                const int32_t slack = static_cast<int32_t>(got)
                                      - static_cast<int32_t>(ToBf16(want));
                const bool bad = (slack > 1 || slack < -1);
                wrong += bad;
                deviating_row[row] += bad;
                if (column < HIDDEN / 2) {
                    deviating_low_half += bad;
                } else {
                    deviating_high_half += bad;
                }
            }
        }
    }
    const int64_t total = experts * ROWS * HIDDEN;
    std::printf("experts %lld  blocks %u  rows a expert %lld\n",
                (long long)experts, blocks, (long long)ROWS);
    std::printf("against the host reference, 1 ulp allowed: %lld of %lld "
                "(%lld untouched)\n", (long long)wrong, (long long)total,
                (long long)untouched);
    for (int64_t row = 0; row < ROWS; ++row) {
        std::printf("  row %lld deviating: %lld of %lld\n", (long long)row,
                    (long long)deviating_row[row],
                    (long long)(experts * HIDDEN));
    }
    std::printf("  columns 0..3071 (subblock 0): %lld of %lld\n"
                "  columns 3072..6143 (subblock 1): %lld of %lld\n",
                (long long)deviating_low_half,
                (long long)(experts * ROWS * HIDDEN / 2),
                (long long)deviating_high_half,
                (long long)(experts * ROWS * HIDDEN / 2));

    for (int round = 0; round < rounds; ++round) {
        timespec begin;
        timespec end;
        clock_gettime(CLOCK_MONOTONIC, &begin);
        for (int step = 0; step < repeats; ++step) {
            routed_int4_launch(blocks, stream, sync_base, rows, stacked,
                               stacked_low, gate_up, (float*)gu_bias, (float*)gu_scale,
                               down, (float*)down_bias, (float*)down_scale,
                               (float*)token, partial, middle,
                               (float*)middle_scale, wide, out, gu_tiling,
                               down_tiling, witness, workspace, experts);
        }
        Check(aclrtSynchronizeStream(stream), "timing");
        clock_gettime(CLOCK_MONOTONIC, &end);
        const double each = Elapsed(begin, end) / repeats;
        std::printf("round %d  %8.3f us for %lld experts, %6.3f us an "
                    "expert\n", round, each, (long long)experts,
                    each / static_cast<double>(experts));
    }

    Check(aclrtDestroyStream(stream), "destroy");
    Check(aclrtResetDevice(device), "reset");
    Check(aclFinalize(), "finalize");
    return (wrong == 0) ? 0 : 1;
}
