/**
 * The cube tiling, in its own translation unit.
 *
 * `matmul_tiling.h` is host-only and pulls in the operator registry's types,
 * so it stays away from the file `bisheng` compiles.
 */
#include <cstring>

#include "front_contract.h"
#include "lib/matmul/matmul_tiling.h"

extern "C" int megakernel_front_build_tiling(int rows, int columns, int width,
                                             void* buffer)
{
    matmul_tiling::MatmulApiTiling tiling;
    tiling.SetAType(matmul_tiling::TPosition::GM,
                    matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_BF16, false);
    tiling.SetBType(matmul_tiling::TPosition::GM,
                    matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_BF16, false);
    tiling.SetCType(matmul_tiling::TPosition::GM,
                    matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_FLOAT);
    tiling.SetShape(rows, columns, megakernel_front::kHidden);
    tiling.SetOrgShape(rows, width, megakernel_front::kHidden);
    tiling.SetBias(false);
    tiling.SetBufferSpace(-1, -1, -1);

    AscendC::tiling::TCubeTiling cube;
    static_assert(sizeof(cube) <= megakernel_front::kTilingStride,
                  "Increase the tiling stride for this CANN version");
    const int64_t status = tiling.GetTiling(cube);
    if (status != 0) {
        return static_cast<int>(status);
    }
    std::memcpy(buffer, &cube, sizeof(cube));
    return 0;
}
